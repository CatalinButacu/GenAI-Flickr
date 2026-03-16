#!/usr/bin/env python3
"""LangGraph-powered long-sequence motion generator with validity enforcement.

Architecture
------------
Actions in this system are **procedurally generated** (not from a dataset).
Each action type (walk, kick, punch, jump, spin, victory, stand) has a
dedicated generator function that produces (N, 21, 3) joint positions using
biomechanically constrained rotations.

The LangGraph pipeline adds a validation-and-repair loop around each segment
to guarantee that every frame in the final output satisfies biomechanical
constraints.  The graph:

::

    ┌─────────────┐
    │  plan_node  │  Parse description → list of action segments
    └──────┬──────┘
           ▼
    ┌──────────────┐
    │ generate_node│  Call procedural generator for current segment
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ validate_node│  Run BodyValidator on all frames
    └──────┬───────┘
           ▼  (violations found AND repair_count < max_repairs)
    ┌──────────────┐
    │  repair_node │  Blend violated frames toward rest pose
    └──────┬───────┘
           ▼  (back to validate)
    ┌──────────────┐
    │  accept_node │  Append segment → collected sequence, advance index
    └──────┬───────┘
           │  more segments remain → back to generate_node
           ▼  all segments done
    ┌──────────────┐
    │  blend_node  │  SLERP transitions between segments (12 frames each)
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ finalize_node│  Save NPY + print report
    └──────────────┘

Usage
-----
::

    python scripts/motion_pipeline.py \\
        --desc "walk forward, kick, punch combo, victory pose" \\
        --fps 24 --out outputs/long_sequence.npy

    # Or from Python:
    from scripts.motion_pipeline import run_pipeline
    result = run_pipeline("walk, spin, jump, stand", fps=24)
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import sys
import time
from typing import Annotated, Sequence

import numpy as np

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.body_validator import (
    BodyValidator,
    SequenceReport,
    repair_frame,
    validate_sequence,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Load procedural generators from demo_cinematic_smpl
# ─────────────────────────────────────────────────────────────────────────────

def _load_demo_module():
    """Import demo_cinematic_smpl.py as a module (avoids __main__ execution)."""
    demo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts", "demo_cinematic_smpl.py",
    )
    spec = importlib.util.spec_from_file_location("demo_cinematic_smpl", demo_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load demo module from {demo_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DEMO = None

def _get_demo():
    global _DEMO
    if _DEMO is None:
        _DEMO = _load_demo_module()
    return _DEMO


# ─────────────────────────────────────────────────────────────────────────────
# Action registry
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: keyword → (generator_fn_name, default_n_frames, aliases)
_ACTION_REGISTRY: list[tuple[str, str, int, list[str]]] = [
    ("walk",    "generate_walk",         72, ["walking", "stride", "march", "step"]),
    ("kick",    "generate_kick",         48, ["kicking", "roundhouse", "sidekick"]),
    ("punch",   "generate_punch_combo",  60, ["punching", "jab", "cross", "hook", "box", "combo"]),
    ("jump",    "generate_jump",         48, ["jumping", "leap", "hop"]),
    ("spin",    "generate_spin",         48, ["spinning", "rotate", "pirouette", "turn", "360"]),
    ("victory", "generate_victory_pose", 48, ["celebrate", "cheer", "win", "arms up", "raise arms"]),
    ("stand",   "_generate_stand",       24, ["standing", "idle", "wait", "rest", "pause"]),
]

# Flat keyword → (generator_fn_name, default_n_frames) lookup
_KEYWORD_MAP: dict[str, tuple[str, int]] = {}
for _name, _fn, _n, _aliases in _ACTION_REGISTRY:
    _KEYWORD_MAP[_name] = (_fn, _n)
    for _alias in _aliases:
        _KEYWORD_MAP[_alias] = (_fn, _n)

# Direction modifier keywords → scalar passed as ``direction`` to generators.
# 1.0 = forward (positive Z),  -1.0 = backward (negative Z).
_DIRECTION_MAP: dict[str, float] = {
    "backward": -1.0, "backwards": -1.0, "back": -1.0,
    "forward"  :  1.0, "forwards"  :  1.0,
}


def call_generator(action: str, n_frames: int, direction: float = 1.0) -> list[np.ndarray]:
    """Call a named procedural generator and return list of (21,3) frames.

    Parameters
    ----------
    action : str
        Action name or alias ('walk', 'kick', 'punch', …).
    n_frames : int
        Number of frames to generate (approximate — generators may vary
        slightly depending on their internal phase logic).
    direction : float
        Travel direction scalar. 1.0 = forward, -1.0 = backward.
        Passed to generators that accept a ``direction`` parameter.

    Returns
    -------
    list of (21, 3) numpy arrays in Y-up mm.
    """
    demo = _get_demo()
    entry = _KEYWORD_MAP.get(action.lower())
    if entry is None:
        log.warning("Unknown action '%s', falling back to stand.", action)
        entry = _KEYWORD_MAP["stand"]
    fn_name, _ = entry
    fn = getattr(demo, fn_name)
    import inspect
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    kwargs: dict = {}
    if params and params[0] in ("n_frames", "n"):
        kwargs[params[0]] = n_frames
    if "direction" in params:
        kwargs["direction"] = direction
    return fn(**kwargs) if kwargs else fn()


# ─────────────────────────────────────────────────────────────────────────────
# Segment description parser
# ─────────────────────────────────────────────────────────────────────────────

# Duration patterns: "for 3 seconds", "3s", "2 sec", "60 frames", "x60"
_DUR_SEC_RE   = re.compile(r"for\s*(\d+(?:\.\d+)?)\s*(?:second|sec|s)\b")
_DUR_FRAME_RE = re.compile(r"(?:for\s*)?(?:x\s*)?(\d+)\s*(?:frame|fr\b)")

DEFAULT_FPS = 24


def parse_description(description: str, fps: int = DEFAULT_FPS) -> list[dict]:
    """Parse a natural-language sequence description into action segments.

    Supports comma/semicolon/then-separated lists with optional durations.

    Examples
    --------
    ::

        "walk for 3 seconds, kick, punch combo, stand for 1 sec, victory"
        "walk x72, spin, jump then stand then victory"

    Returns
    -------
    list of dicts: [{"action": str, "n_frames": int, "fn": str}, ...]
    """
    # Split on commas, semicolons, "then", "and then", "followed by"
    parts = re.split(r",|;|\s+then\s+|\s+and\s+then\s+|\s+followed\s+by\s+", description, flags=re.I)
    segments = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract duration
        n_frames: int | None = None
        m = _DUR_SEC_RE.search(part)
        if m:
            n_frames = max(1, int(float(m.group(1)) * fps))
            part = part[:m.start()] + part[m.end():]
        else:
            m = _DUR_FRAME_RE.search(part)
            if m:
                n_frames = int(m.group(1))
                part = part[:m.start()] + part[m.end():]
        part = part.strip(" ,;")

        # Match action keyword (longest-match)
        matched_key = None
        matched_len = 0
        for kw in _KEYWORD_MAP:
            if kw in part.lower() and len(kw) > matched_len:
                matched_key = kw
                matched_len = len(kw)

        if matched_key is None:
            log.warning("Could not parse action from '%s', interpreting as 'stand'.", part)
            matched_key = "stand"

        fn_name, default_n = _KEYWORD_MAP[matched_key]
        # Detect travel-direction modifiers (e.g. "backward", "forward")
        direction = 1.0
        for dk, dval in _DIRECTION_MAP.items():
            if re.search(r"\b" + dk + r"\b", part.lower()):
                direction = dval
                break
        segments.append({
            "action":    matched_key,
            "fn":        fn_name,
            "n_frames":  n_frames if n_frames is not None else default_n,
            "label":     part or matched_key,
            "direction": direction,
        })

    if not segments:
        log.warning("No actions parsed; defaulting to stand(24)+walk(72)+stand(24).")
        segments = [
            {"action": "stand",  "fn": "_generate_stand",        "n_frames": 24, "label": "stand"},
            {"action": "walk",   "fn": "generate_walk",           "n_frames": 72, "label": "walk"},
            {"action": "stand",  "fn": "_generate_stand",         "n_frames": 24, "label": "stand"},
        ]

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Transition blending
# ─────────────────────────────────────────────────────────────────────────────

def blend_transition(
    from_frame: np.ndarray,
    to_frame: np.ndarray,
    n_blend: int = 12,
) -> list[np.ndarray]:
    """Generate *n_blend* interpolation frames between two skeleton poses.

    Uses linear interpolation on joint positions then re-projects each
    blended frame through ``_enforce_bone_lengths`` (from the demo module)
    so that arm/leg segments don't shrink during the transition.

    The start and end frames are NOT included.
    """
    enforce = getattr(_get_demo(), "_enforce_bone_lengths", None)
    frames = []
    for i in range(1, n_blend + 1):
        t = i / (n_blend + 1)
        interp = (1.0 - t) * from_frame + t * to_frame
        if enforce is not None:
            enforce(interp)  # in-place
        frames.append(interp)
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Input ────────────────────────────────────────────────────
    description:        str
    fps:                int
    max_repairs:        int
    n_blend_frames:     int
    output_path:        str

    # ── Plan ─────────────────────────────────────────────────────
    plan:               list[dict]   # [{action, fn, n_frames, label}, …]
    segment_idx:        int          # which segment is being processed

    # ── Current segment ──────────────────────────────────────────
    current_frames:     list         # list of (21,3) arrays
    repair_count:       int
    current_valid:      bool
    current_summary:    str          # human-readable validation note

    # ── Accumulated result ───────────────────────────────────────
    accepted_segments:  list         # list of list[np.ndarray]

    # ── Final output ─────────────────────────────────────────────
    final_sequence:     list         # flat list of (21,3) arrays
    final_report:       str          # printed summary


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR = BodyValidator()


def plan_node(state: PipelineState) -> PipelineState:
    """Parse the text description into a list of action segments."""
    plan = parse_description(state["description"], fps=state["fps"])
    log.info("[plan_node] %d segments: %s",
             len(plan), [f"{s['label']}({s['n_frames']}fr)" for s in plan])
    return {**state, "plan": plan, "segment_idx": 0, "accepted_segments": []}


def generate_node(state: PipelineState) -> PipelineState:
    """Generate skeleton frames for the current segment."""
    seg = state["plan"][state["segment_idx"]]
    log.info("[generate_node] Segment %d/%d: %s (%d frames)",
             state["segment_idx"] + 1, len(state["plan"]),
             seg["label"], seg["n_frames"])

    direction = seg.get("direction", 1.0)
    log.info("[generate_node]   direction=%.1f", direction)
    frames = call_generator(seg["action"], seg["n_frames"], direction=direction)
    log.info("[generate_node]   → %d frames produced", len(frames))

    return {
        **state,
        "current_frames": frames,
        "repair_count":   0,
        "current_valid":  False,
        "current_summary": "",
    }


def validate_node(state: PipelineState) -> PipelineState:
    """Validate all frames in the current segment."""
    frames = state["current_frames"]
    report = _VALIDATOR.evaluate_sequence(frames, verbose=False)

    is_valid = report.valid
    pct = report.validity_rate * 100
    vcount = report.invalid_frames

    summary = (
        f"Segment {state['segment_idx']+1}: "
        f"{pct:.1f}% valid ({len(frames)-vcount}/{len(frames)} frames OK) "
        f"repairs={state['repair_count']}"
    )
    log.info("[validate_node] %s", summary)

    return {
        **state,
        "current_valid":   is_valid,
        "current_summary": summary,
    }


def repair_node(state: PipelineState) -> PipelineState:
    """Repair invalid frames by blending them toward rest pose."""
    frames = [f.copy() for f in state["current_frames"]]
    repaired_count = 0

    for i, frame in enumerate(frames):
        frame_report = _VALIDATOR.evaluate_frame(frame, frame_idx=i)
        if not frame_report.valid:
            frames[i] = repair_frame(frame, frame_report.violations)
            repaired_count += 1

    rc = state["repair_count"] + 1
    log.info("[repair_node]   Repaired %d frame(s) (pass %d)", repaired_count, rc)

    return {**state, "current_frames": frames, "repair_count": rc}


def accept_node(state: PipelineState) -> PipelineState:
    """Accept the current segment and advance to the next."""
    segs = list(state["accepted_segments"]) + [list(state["current_frames"])]
    nxt  = state["segment_idx"] + 1
    log.info("[accept_node]  Segment %d accepted (%s)",
             state["segment_idx"] + 1, state["current_summary"])
    return {
        **state,
        "accepted_segments": segs,
        "segment_idx":       nxt,
        "current_frames":    [],
    }


def blend_node(state: PipelineState) -> PipelineState:
    """Stitch accepted segments with smooth transition frames."""
    segments = state["accepted_segments"]
    n_blend  = state["n_blend_frames"]

    if not segments:
        return {**state, "final_sequence": []}

    full: list[np.ndarray] = list(segments[0])
    for seg in segments[1:]:
        if full and seg:
            trans = blend_transition(full[-1], seg[0], n_blend=n_blend)
            full.extend(trans)
        full.extend(seg)

    log.info("[blend_node]   Total frames after blending: %d", len(full))
    return {**state, "final_sequence": full}


def finalize_node(state: PipelineState) -> PipelineState:
    """Validate the complete sequence, save to file, and print final report."""
    final = state["final_sequence"]
    out   = state["output_path"]

    # Final full-sequence validation
    log.info("[finalize_node] Running final validation on %d frames…", len(final))
    report = validate_sequence(final, verbose=False)
    report.print_summary()

    # Save NPY
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    arr = np.stack(final, axis=0)   # (N, 21, 3)
    np.save(out, arr)
    log.info("[finalize_node] Saved sequence → %s  shape=%s", out, arr.shape)

    summary = (
        f"Saved {arr.shape[0]} frames to '{out}'.  "
        f"Validity: {report.validity_rate*100:.1f}%"
    )
    return {**state, "final_report": summary}


# ─────────────────────────────────────────────────────────────────────────────
# Conditional routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_validate(state: PipelineState) -> str:
    """After validation: repair if invalid + under repair budget, else accept."""
    if not state["current_valid"] and state["repair_count"] < state["max_repairs"]:
        return "repair"
    return "accept"


def _route_after_accept(state: PipelineState) -> str:
    """After accepting a segment: generate next or blend if all done."""
    if state["segment_idx"] < len(state["plan"]):
        return "generate"
    return "blend"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> CompiledStateGraph:
    """Build and compile the motion pipeline LangGraph."""
    g = StateGraph(PipelineState)

    # Add all nodes
    g.add_node("plan",     plan_node)
    g.add_node("generate", generate_node)
    g.add_node("validate", validate_node)
    g.add_node("repair",   repair_node)
    g.add_node("accept",   accept_node)
    g.add_node("blend",    blend_node)
    g.add_node("finalize", finalize_node)

    # Edges
    g.set_entry_point("plan")
    g.add_edge("plan",     "generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"repair": "repair", "accept": "accept"},
    )
    g.add_edge("repair", "validate")
    g.add_conditional_edges(
        "accept",
        _route_after_accept,
        {"generate": "generate", "blend": "blend"},
    )
    g.add_edge("blend",    "finalize")
    g.add_edge("finalize", END)

    return g.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    description: str,
    fps: int = 24,
    max_repairs: int = 3,
    n_blend_frames: int = 12,
    output_path: str = "outputs/long_sequence.npy",
) -> PipelineState:
    """Run the full motion generation + validation pipeline.

    Parameters
    ----------
    description : str
        Natural-language action description.
        e.g. ``"walk for 3 seconds, kick, punch combo, victory"``
    fps : int
        Frames per second (used to convert second durations to frames).
    max_repairs : int
        Maximum repair passes per segment before accepting anyway.
    n_blend_frames : int
        Number of interpolation frames between consecutive segments.
    output_path : str
        Output NPY file path.

    Returns
    -------
    Final :class:`PipelineState` dict.
    """
    graph = build_graph()

    initial: PipelineState = {
        "description":     description,
        "fps":             fps,
        "max_repairs":     max_repairs,
        "n_blend_frames":  n_blend_frames,
        "output_path":     output_path,
        "plan":            [],
        "segment_idx":     0,
        "current_frames":  [],
        "repair_count":    0,
        "current_valid":   False,
        "current_summary": "",
        "accepted_segments": [],
        "final_sequence":  [],
        "final_report":    "",
    }

    t0 = time.time()
    result: PipelineState = graph.invoke(initial)  # type: ignore[assignment]
    elapsed = time.time() - t0
    log.info("Pipeline completed in %.1f s.  %s", elapsed, result.get("final_report", ""))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate and validate long motion sequences via LangGraph."
    )
    ap.add_argument(
        "--desc", "-d",
        default="stand for 1 second, walk for 3 seconds, kick, punch combo, jump, stand for 1 second, victory",
        help="Natural-language description of the motion sequence.",
    )
    ap.add_argument("--fps",          type=int,   default=24,
                    help="Frames per second (default 24).")
    ap.add_argument("--max-repairs",  type=int,   default=3,
                    help="Max repair passes per segment (default 3).")
    ap.add_argument("--blend",        type=int,   default=12,
                    help="Blend frames between segments (default 12 ≈ 0.5 s).")
    ap.add_argument("--out",          type=str,   default="outputs/long_sequence.npy",
                    help="Output NPY path.")
    ap.add_argument("--validate-only", type=str,  default=None,
                    help="Skip generation; validate an existing NPY file instead.")
    args = ap.parse_args()

    if args.validate_only:
        # Standalone validation mode
        data = np.load(args.validate_only)
        if data.ndim == 2:
            data = data[np.newaxis]
        frames = [data[i] for i in range(data.shape[0])]
        print(f"Validating {len(frames)} frames from '{args.validate_only}'…")
        validate_sequence(frames, verbose=True)
    else:
        print(f"\nDescription: {args.desc!r}")
        print(f"FPS={args.fps}  max_repairs={args.max_repairs}  "
              f"blend={args.blend}fr  out={args.out}\n")
        run_pipeline(
            description=args.desc,
            fps=args.fps,
            max_repairs=args.max_repairs,
            n_blend_frames=args.blend,
            output_path=args.out,
        )

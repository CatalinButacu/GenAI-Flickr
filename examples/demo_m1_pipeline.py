#!/usr/bin/env python3
"""
GenAI-Flickr  --  M1 Pipeline Demo
====================================
Text prompt  →  Scene Extraction (M1)  →  Physics Simulation  →  Video

Usage
-----
    python examples/demo_m1_pipeline.py
    python examples/demo_m1_pipeline.py --prompt "a red ball on a wooden table"
    python examples/demo_m1_pipeline.py --prompt "a dog chases a cat in a park" \\
                                         --output outputs/videos/demo_m1.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Colour helpers ────────────────────────────────────────────────────────────
R  = "\033[91m"
G  = "\033[92m"
Y  = "\033[93m"
B  = "\033[94m"
M  = "\033[95m"
C  = "\033[96m"
W  = "\033[97m"
DIM= "\033[2m"
BO = "\033[1m"
RST= "\033[0m"
LINE = "─" * 68


def banner():
    print(f"\n{BO}{C}")
    print("  +" + "=" * 62 + "+")
    print("  |           GenAI-Flickr  -  M1 Pipeline Demo                 |")
    print("  |   Text  ->  Scene Graph  ->  Physics Sim  ->  Video         |")
    print("  +" + "=" * 62 + "+")
    print(RST)


def step(n: int, title: str):
    print(f"\n{BO}{B}  [{n}] {title}{RST}")
    print(f"  {DIM}{LINE}{RST}")


def ok(msg: str):   print(f"  {G}[OK]{RST}  {msg}")
def info(msg: str): print(f"  {DIM}      {msg}{RST}")
def warn(msg: str): print(f"  {Y}[WARN]{RST}  {msg}")


# ── M1 inference ─────────────────────────────────────────────────────────────

def _find_checkpoint() -> str:
    """Return best available M1 checkpoint, newest first."""
    candidates = [
        "m1_checkpoints/m1_scene_extractor_v5",
        "m1_checkpoints/m1_scene_extractor_v4",
        "m1_checkpoints/m1_scene_extractor_v3",
        "m1_checkpoints/m1_scene_extractor",
    ]
    for c in candidates:
        if (Path(c) / "config.json").exists():
            return c
    raise FileNotFoundError("No M1 checkpoint found. Run scripts/train_m1_t5.py first.")


def _postprocess(raw: str) -> str:
    """Reverse <extra_id_0>/<extra_id_1> → { / } and strip padding tokens."""
    raw = raw.replace("<extra_id_0>", "{").replace("<extra_id_1>", "}")
    raw = raw.replace("</s>", "").replace("<pad>", "").replace("<s>", "")
    raw = re.sub(r"<extra_id_\d+>", "", raw)
    return raw.strip()


def _repair(raw: str) -> str:
    """
    Best-effort repair of the common flan-T5-small JSON corruption patterns:
    - stray } before a string value:  "key": } "val"  →  "key": "val"
    - single-quote open:              "key": 'val"    →  "key": "val"
    - run-on word value:              "key": val"     →  "key": "val"
    - missing colon:                  "key" "val"     →  "key": "val"
    - <unk> tokens
    """
    s = raw.replace("<unk>", "")

    # stray } or , before a string value
    s = re.sub(r':\s*[},]+\s*"', ': "', s)
    # single-quote open → double-quote
    s = re.sub(r":\s*'([^\"]+\")", r': "\1', s)
    # bare word value missing opening quote:  : word"  →  : "word"
    s = re.sub(r':\s*([A-Za-z][A-Za-z0-9 _-]*)"', r': "\1"', s)
    # malformed key with space instead of colon:  "key "val"  →  "key": "val"
    s = re.sub(r'"([^"]+)\s+"([^"]+)"', r'"\1": "\2"', s)
    # remove trailing , before ] or }
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # ensure outer braces
    if not s.startswith("{"):
        s = "{" + s
    if not s.endswith("}"):
        s += "}"
    return s


def run_m1(prompt: str, checkpoint: str) -> tuple[dict | None, str]:
    """Run M1 inference and return parsed scene graph (best effort)."""
    import torch
    from transformers import T5ForConditionalGeneration, AutoTokenizer

    tok   = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    model.eval()

    ids = tok(f"extract scene: {prompt}", max_length=256,
              truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_length=512, num_beams=4,
                              early_stopping=True)

    raw = tok.decode(out[0], skip_special_tokens=False)
    raw = _postprocess(raw)

    for candidate in (raw, _repair(raw)):
        try:
            return json.loads(candidate), raw
        except json.JSONDecodeError:
            pass

    # Final fallback: regex-extract names and build minimal dict
    names = re.findall(r'"name":\s*["\']?([A-Za-z][A-Za-z0-9 _-]*)"', raw)
    preds = re.findall(r'"predicate":\s*"([^"]+)"', raw)
    if names:
        entities = [{"id": f"obj_{i}", "name": n, "type": "object"}
                    for i, n in enumerate(names)]
        relations = [{"subject": names[0], "predicate": p,
                       "object": names[min(1, len(names)-1)]} for p in preds]
        return {"entities": entities, "actions": [], "relations": relations}, raw
    return None, raw


def display_scene(data: dict):
    entities  = data.get("entities",  [])
    actions   = data.get("actions",   [])
    relations = data.get("relations", [])

    print(f"\n  {BO}Entities ({len(entities)}){RST}")
    for e in entities:
        name  = e.get("name", "?")
        etype = e.get("type", "object")
        attrs = e.get("attributes", {})
        color = attrs.get("color", "") if isinstance(attrs, dict) else ""
        col   = f"  {DIM}color={color}{RST}" if color else ""
        print(f"    {G}*{RST} {BO}{name}{RST}  {DIM}[{etype}]{RST}{col}")

    if relations:
        print(f"\n  {BO}Relations ({len(relations)}){RST}")
        for r in relations:
            print(f"    {DIM}{r.get('subject','?')}{RST}  {C}--[{r.get('predicate','?')}]-->{RST}  {r.get('object','?')}")

    if actions:
        print(f"\n  {BO}Actions ({len(actions)}){RST}")
        for a in actions:
            print(f"    {M}>>{RST} {a.get('verb','?')}  actor={a.get('actor','?')}")


# ── Entity → Physics shape mapper ────────────────────────────────────────────

_SHAPE_MAP = {
    # spheres
    "ball":    "sphere", "sphere": "sphere", "apple": "sphere",
    "orange":  "sphere", "head":   "sphere", "globe": "sphere",
    # cylinders (engine supports: box / sphere / cylinder only)
    "bottle":  "cylinder", "can":  "cylinder", "cup":   "cylinder",
    "mug":     "cylinder", "glass":"cylinder", "barrel":"cylinder",
    # characters / animals → cylinder (upright approximation)
    "person":  "cylinder", "man":    "cylinder", "woman": "cylinder",
    "child":   "cylinder", "robot":  "cylinder", "dog":   "cylinder",
    "cat":     "cylinder", "bear":   "cylinder", "animal":"cylinder",
    "horse":   "cylinder", "bird":   "cylinder", "paws":  "cylinder",
}
_COLOR_MAP = {
    "red":    [0.9, 0.1, 0.1, 1.0], "blue":  [0.1, 0.3, 0.9, 1.0],
    "green":  [0.1, 0.8, 0.1, 1.0], "yellow":[0.9, 0.85, 0.0, 1.0],
    "orange": [0.9, 0.5, 0.0, 1.0], "brown": [0.55, 0.27, 0.07, 1.0],
    "white":  [0.95, 0.95, 0.95, 1.0], "black":[0.1, 0.1, 0.1, 1.0],
    "grey":   [0.5, 0.5, 0.5, 1.0],   "gray": [0.5, 0.5, 0.5, 1.0],
    "purple": [0.5, 0.0, 0.8, 1.0],   "pink": [0.95, 0.4, 0.7, 1.0],
    "wooden": [0.55, 0.35, 0.15, 1.0],"metal":[0.6, 0.6, 0.65, 1.0],
}
_DEFAULT_COLORS = [
    [0.2, 0.6, 0.9, 1.0], [0.9, 0.3, 0.2, 1.0], [0.2, 0.8, 0.3, 1.0],
    [0.9, 0.7, 0.1, 1.0], [0.7, 0.2, 0.9, 1.0], [0.1, 0.8, 0.8, 1.0],
]


def _entity_color(entity: dict, idx: int) -> list:
    attrs = entity.get("attributes", {})
    if isinstance(attrs, dict):
        for k in ("color", "material"):
            val = attrs.get(k, "").lower()
            for cname, rgba in _COLOR_MAP.items():
                if cname in val:
                    return rgba
    name = entity.get("name", "").lower()
    for cname, rgba in _COLOR_MAP.items():
        if cname in name:
            return rgba
    return _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]


def _entity_shape(entity: dict) -> str:
    name = entity.get("name", "").lower()
    for keyword, shape in _SHAPE_MAP.items():
        if keyword in name:
            return shape
    return "box"


def _entity_size(shape: str) -> list:
    return {"sphere": [0.15], "cylinder": [0.1, 0.35],
            "box": [0.18, 0.18, 0.18]}.get(shape, [0.18, 0.18, 0.18])


# ── Physics simulation ────────────────────────────────────────────────────────

def run_physics(entities: list, output_path: str) -> Optional[str]:
    try:
        from src.modules.m5_physics_engine import Scene, Simulator, CameraConfig
    except ImportError:
        warn("Physics engine not available (pybullet missing?).")
        return None

    scene = Scene(gravity=-9.81)
    scene.setup()
    scene.add_ground()

    cols = 3
    for idx, ent in enumerate(entities[:6]):     # max 6 objects
        shape = _entity_shape(ent)
        size  = _entity_size(shape)
        color = _entity_color(ent, idx)
        row, col = divmod(idx, cols)
        x = (col - 1) * 0.5
        y = (row)  * 0.5
        z = 0.5 + idx * 0.15                     # stagger heights slightly
        scene.add_primitive(
            name     = f"obj_{idx}",
            shape    = shape,
            size     = size,
            mass     = 0.5 if shape == "sphere" else 1.0,
            position = [x, y, z],
            color    = color,
        )
        ok(f"Spawned  {BO}{ent.get('name')}{RST}  as {shape}  @ ({x:.1f}, {y:.1f}, {z:.1f})")

    camera = CameraConfig(
        width=960, height=540,
        target=[0, 0, 0.3], distance=3.0, yaw=35, pitch=-22,
    )
    sim = Simulator(scene, camera)

    # Give a random horizontal nudge to the first sphere/ball
    actions = []
    for idx, ent in enumerate(entities[:6]):
        if _entity_shape(ent) == "sphere":
            actions.append({
                "time": 0.2,
                "object": f"obj_{idx}",
                "type": "velocity",
                "velocity": [1.5, 0.5, 0.3],
            })
            break

    frames = sim.run(duration=4.0, fps=30, actions=actions)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    video = sim.create_video(frames=frames, output_path=output_path,
                             fps=30, layout="horizontal")
    return video


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="GenAI-Flickr M1 Pipeline Demo")
    ap.add_argument("--prompt",  default="a red ball falls onto a wooden table next to a blue cup")
    ap.add_argument("--output",  default="outputs/videos/demo_m1_pipeline.mp4")
    ap.add_argument("--checkpoint", default=None,
                    help="Override M1 checkpoint path")
    ap.add_argument("--no-physics", action="store_true",
                    help="Skip physics simulation (just show scene extraction)")
    return ap.parse_args()


def _step1_load_model(args) -> str:
    """Step 1: Find checkpoint and show info. Returns checkpoint path."""
    step(1, "Loading M1 Scene Extractor")
    checkpoint = args.checkpoint or _find_checkpoint()
    meta = Path(checkpoint) / "training_metadata.json"
    loss_str = ""
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            el = m.get("final_metrics", {}).get("eval_loss")
            ep = m.get("epochs")
            loss_str = f"  eval_loss={el:.4f}  epochs={ep}" if el else ""
        except Exception:
            pass
    ok(f"Checkpoint: {BO}{checkpoint}{RST}{DIM}{loss_str}{RST}")
    return checkpoint


def _supplement_entities(entities: list, prompt: str) -> list:
    """Supplement entities with keyword-matched objects from the prompt."""
    known = list(_SHAPE_MAP.keys())
    words = re.findall(r'\b[a-zA-Z]+\b', prompt.lower())
    seen_names = {e.get("name", "").lower() for e in entities}
    original_count = len(entities)
    for w in words:
        if w in known and w not in seen_names and len(entities) < 6:
            entities.append({"id": f"obj_kw_{w}", "name": w, "type": "object"})
            seen_names.add(w)
    if len(entities) > original_count:
        info(f"Supplemented with {len(entities) - original_count} keyword-matched entit(ies) from prompt")
    return entities


def main():
    args = parse_args()
    banner()

    checkpoint = _step1_load_model(args)

    # ── Step 2: Run M1 ────────────────────────────────────────────────────
    step(2, "Scene Extraction  (M1 · flan-T5-small)")
    print(f"\n  {BO}Prompt:{RST}  {Y}\"{args.prompt}\"{RST}\n")
    info("Running inference ...")
    t0 = time.time()
    data, raw = run_m1(args.prompt, checkpoint)
    elapsed = time.time() - t0

    if data is None:
        warn("Could not extract entities from this prompt.")
        warn(f"Raw output: {raw[:200]}")
        sys.exit(1)

    ok(f"Extracted in {elapsed:.2f}s")

    # Supplement: if M1 returned <2 entities, scan the prompt directly for
    # known object keywords so the physics demo stays interesting
    entities = data.get("entities", [])
    if len(entities) < 2:
        entities = _supplement_entities(entities, args.prompt)
        data["entities"] = entities

    display_scene(data)

    entities = data.get("entities", [])
    if not entities:
        warn("No entities found — cannot spawn physics objects.")
        sys.exit(0)

    if args.no_physics:
        print(f"\n  {DIM}--no-physics flag set, skipping simulation.{RST}")
        sys.exit(0)

    # ── Step 3: Physics ───────────────────────────────────────────────────
    step(3, "Physics Simulation  (M5 · PyBullet)")
    info(f"Spawning {min(len(entities), 6)} object(s) ...")
    print()

    video = run_physics(entities, args.output)

    # ── Step 4: Done ──────────────────────────────────────────────────────
    step(4, "Done")
    if video and Path(video).exists():
        size_mb = Path(video).stat().st_size / 1e6
        ok(f"Video saved -> {BO}{video}{RST}  ({size_mb:.1f} MB)")
    else:
        warn("Physics simulation did not produce a video.")

    print(f"\n  {DIM}Pipeline: prompt → M1 scene graph → M5 physics → video{RST}")
    print(f"  {DIM}Checkpoint used: {checkpoint}{RST}\n")


if __name__ == "__main__":
    main()

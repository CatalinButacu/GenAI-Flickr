"""
M1 CLI Test Tool  (Single Responsibility: Interactive Prompt Testing)
=====================================================================
Terminal interface for testing M1 end-to-end.

Displays:
  [DIRECT]   — entities/actions/relations extracted from the raw text
  [INDIRECT] — enriched physics, inferred templates, gaps

Usage:
    python scripts/test_m1.py --prompt "a red ball falls on a wooden table"
    python scripts/test_m1.py                   # interactive REPL
    python scripts/test_m1.py --mode rules      # force rule-based only
    python scripts/test_m1.py --mode ml         # force ML only (needs checkpoint)
    python scripts/test_m1.py --json            # emit raw JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

# Ensure the project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.modules.scene_understanding.orchestrator import StoryAgent
from src.modules.scene_understanding.retriever import KnowledgeRetriever
from src.modules.scene_understanding.processor import NLPProcessor
from src.modules.scene_understanding.extractor import RuleExtractor
from src.modules.scene_understanding.ml_extractor import MLExtractor
from src.modules.scene_understanding.reasoner import Reasoner
from src.modules.scene_understanding.models import ExtractionResult, EntityType


# ---------------------------------------------------------------------------
# ANSI colours (graceful fallback on Windows without ANSI support)
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class Color:
    RESET  = "\033[0m"  if _supports_color() else ""
    BOLD   = "\033[1m"  if _supports_color() else ""
    CYAN   = "\033[96m" if _supports_color() else ""
    GREEN  = "\033[92m" if _supports_color() else ""
    YELLOW = "\033[93m" if _supports_color() else ""
    RED    = "\033[91m" if _supports_color() else ""
    GRAY   = "\033[90m" if _supports_color() else ""
    BLUE   = "\033[94m" if _supports_color() else ""


# ---------------------------------------------------------------------------
# Output formatter  (SRP: "pretty-print M1 results to terminal")
# ---------------------------------------------------------------------------

class M1OutputFormatter:
    WIDTH = 68

    def header(self, title: str) -> None:
        bar = "═" * self.WIDTH
        print(f"\n{Color.CYAN}{Color.BOLD}{bar}{Color.RESET}")
        print(f"{Color.CYAN}{Color.BOLD}  {title}{Color.RESET}")
        print(f"{Color.CYAN}{bar}{Color.RESET}")

    def section(self, title: str) -> None:
        print(f"\n{Color.BOLD}{Color.YELLOW}{'━' * 4} {title} {'━' * (self.WIDTH - len(title) - 6)}{Color.RESET}")

    def kv(self, key: str, value: str, indent: int = 2) -> None:
        pad = " " * indent
        print(f"{pad}{Color.GRAY}{key:<18}{Color.RESET}{value}")

    def bullet(self, text: str, indent: int = 4, color: str = "") -> None:
        pad = " " * indent
        print(f"{pad}{color}•{Color.RESET} {text}")

    def print_prompt(self, prompt: str, mode: str) -> None:
        self.kv("PROMPT", f'"{prompt[:80]}"')
        self.kv("EXTRACTION MODE", mode)

    def print_direct(self, raw: ExtractionResult) -> None:
        self.section("DIRECT EXTRACTIONS — from raw text")

        # Entities
        print(f"  {Color.BOLD}Entities  ({len(raw.entities)}){Color.RESET}")
        for e in raw.entities:
            type_name = e.entity_type.name.lower()
            count_str = f"×{e.count}" if e.count > 1 else ""
            static    = " [static]" if e.is_static else ""
            attrs     = ""
            color_v   = e.get_attr("color_vec")
            if color_v:
                attrs += f"  color={color_v}"
            self.bullet(
                f"{Color.GREEN}{e.name}{Color.RESET}{count_str}  "
                f"type={type_name}"
                f"  mass={e.mass or '?'}kg"
                f"  dims={e.dimensions or '?'}"
                f"{static}{attrs}",
                color=Color.GREEN
            )

        # Actions
        print(f"\n  {Color.BOLD}Actions   ({len(raw.actions)}){Color.RESET}")
        for a in raw.actions:
            target = f" → {a.target_id}" if a.target_id else ""
            self.bullet(
                f"{Color.BLUE}{a.verb}{Color.RESET}  actor={a.actor_id or '?'}{target}  "
                f"params={a.parameters}",
                color=Color.BLUE
            )

        # Relations
        print(f"\n  {Color.BOLD}Relations ({len(raw.relations)}){Color.RESET}")
        for r in raw.relations:
            self.bullet(
                f"{r.source_id} {Color.CYAN}--[{r.relation}]-->{Color.RESET} {r.target_id}  "
                f"conf={r.confidence:.2f}",
                color=Color.CYAN
            )

    def print_indirect(self, enriched: ExtractionResult) -> None:
        self.section("INDIRECT EXTRACTIONS — inferred / enriched")

        # Physics enrichment
        print(f"  {Color.BOLD}Physics enrichment{Color.RESET}")
        for e in enriched.entities:
            parts = []
            if e.mass:
                parts.append(f"mass={e.mass:.2f}kg")
            if e.dimensions:
                d = e.dimensions
                dims_str = "×".join(f"{v:.2f}" for v in d.values())
                parts.append(f"dims={dims_str}m")
            if e.material:
                parts.append(f"material={e.material}")
            if parts:
                self.bullet(
                    f"{Color.GREEN}{e.name}{Color.RESET}  " + "  ".join(parts),
                    color=Color.GREEN
                )

        # Template match
        if enriched.inferred_activity:
            print(f"\n  {Color.BOLD}Template{Color.RESET}  matched={enriched.inferred_activity}")
        else:
            print(f"\n  {Color.BOLD}Template{Color.RESET}  none matched")

        if enriched.inferred_setting:
            print(f"  {Color.BOLD}Setting{Color.RESET}   inferred={enriched.inferred_setting}")

        print(f"\n  {Color.BOLD}Fields inferred{Color.RESET}  (gaps removed — use reasoner logs for detail)")

    def print_pipeline_output(self, scene, duration: float) -> None:
        self.section("PIPELINE OUTPUT — SceneDescription")
        self.kv("Objects",  str(len(scene.objects)))
        self.kv("Actions",  str(len(scene.actions)))
        self.kv("Duration", f"{duration:.1f}s")
        self.kv("Style",    scene.style_prompt[:60])

    def footer(self) -> None:
        print(f"\n{Color.GRAY}{'─' * self.WIDTH}{Color.RESET}\n")


# ---------------------------------------------------------------------------
# M1 runner  (SRP: "execute M1 and surface intermediate results")
# ---------------------------------------------------------------------------

class M1Runner:
    """
    Runs M1 pipeline components step-by-step so the CLI can inspect
    direct (pre-reasoning) and indirect (post-reasoning) outputs.
    """

    def __init__(self, mode: str = "auto") -> None:
        self._mode      = mode
        self._agent     = StoryAgent()
        self._processor = NLPProcessor()
        self._retriever = KnowledgeRetriever()
        self._reasoner  = Reasoner(retriever=self._retriever)
        self._builder   = __import__(
            'src.modules.scene_understanding.builder', fromlist=['SceneBuilder']
        ).SceneBuilder()
        self._setup = False

    def setup(self) -> None:
        self._agent.setup()
        self._retriever.setup()
        self._setup = True

    @property
    def active_mode(self) -> str:
        if self._mode == "rules":  return "rules (forced)"
        if self._mode == "ml":    return "ml (forced)"
        return f"{self._agent.extraction_mode} (cascade)"

    def run(self, prompt: str) -> tuple:
        """Returns (raw_extraction, enriched_extraction, scene_description)."""
        # --- Direct (raw extraction) ---
        if self._mode == "rules":
            raw = RuleExtractor().extract(prompt, retriever=self._retriever)
        elif self._mode == "ml":
            ml = MLExtractor()
            ml.load()
            raw = ml.extract(prompt) if ml.is_loaded else self._processor.process(prompt, self._retriever)
        else:
            raw = self._processor.process(prompt, retriever=self._retriever)

        # --- Indirect (reasoning enrichment) ---
        enriched = self._reasoner.reason(raw)

        # --- Pipeline output (SceneDescription directly from builder) ---
        scene = self._builder.build(enriched)

        return raw, enriched, scene


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M1 Scene Understanding — interactive CLI tester")
    p.add_argument("--prompt", "-p", default=None, help="Single prompt to test (else REPL mode)")
    p.add_argument("--mode",   choices=["auto", "rules", "ml"], default="auto", help="Extraction mode: auto (cascade), rules, ml")
    p.add_argument("--json",   action="store_true", help="Emit raw JSON instead of pretty output")
    return p.parse_args()


def _run_once(runner: M1Runner, formatter: M1OutputFormatter, prompt: str, emit_json: bool) -> None:
    raw, enriched, scene = runner.run(prompt)

    if emit_json:
        import dataclasses
        print(json.dumps({
            "prompt": prompt,
            "mode":   runner.active_mode,
            "direct": {
                "entities":  [e.name for e in raw.entities],
                "actions":   [a.verb for a in raw.actions],
                "relations": [{"s": r.source_id, "rel": r.relation, "o": r.target_id}
                              for r in raw.relations],
            },
            "indirect": {
                "inferred_activity": enriched.inferred_activity,
                "inferred_setting":  enriched.inferred_setting,
            },
            "pipeline": {
                "objects":  len(scene.objects),
                "actions":  len(scene.actions),
                "duration": scene.duration,
            },
        }, indent=2))
        return

    formatter.header("M1 — Scene Understanding Output")
    formatter.print_prompt(prompt, runner.active_mode)
    formatter.print_direct(raw)
    formatter.print_indirect(enriched)
    formatter.print_pipeline_output(scene, scene.duration)
    formatter.footer()


def main() -> None:
    args = _parse_args()

    runner    = M1Runner(mode=args.mode)
    formatter = M1OutputFormatter()

    print(f"{Color.CYAN}Setting up M1 …{Color.RESET}", end=" ", flush=True)
    runner.setup()
    print(f"{Color.GREEN}ready ({runner.active_mode}){Color.RESET}")

    if args.prompt:
        _run_once(runner, formatter, args.prompt, args.json)
    else:
        # REPL mode
        print(f"\n{Color.BOLD}M1 REPL{Color.RESET}  — type a scene prompt, or 'quit' to exit.\n")
        while True:
            try:
                prompt = input(f"{Color.YELLOW}> {Color.RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit", "q"}:
                break
            _run_once(runner, formatter, prompt, args.json)


if __name__ == "__main__":
    main()

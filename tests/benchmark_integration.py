"""
Integration Benchmark — End-to-End Pipeline
============================================
Tests the full pipeline (M1→M2→M4→M5→M7→MP4) on diverse prompts.

Validates:
  - Pipeline produces video files
  - Parsed scenes have entities and actions
  - Physics frames are generated
  - Videos have non-zero file size

Run: py tests/benchmark_integration.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Test prompts (diverse coverage)
# ---------------------------------------------------------------------------

_PROMPTS = [
    # Simple single-entity
    ("single_walk",   "a person walks forward"),
    ("single_run",    "a man runs quickly"),
    ("single_stand",  "a woman stands still"),
    # Multi-entity with relations
    ("two_objects",   "a red ball on a blue table"),
    ("person_object", "a person kicks a ball"),
    # Spatial + action combos
    ("near",          "a cat near a dog"),
    ("fall",          "a box falls on a table"),
    # Complex scenes
    ("three_entity",  "a person walks beside a tree next to a house"),
    ("action_seq",    "a man jumps and then waves"),
    # Edge: minimal
    ("minimal",       "a ball"),
]


class IntegrationBenchmark:
    """Run E2E pipeline on diverse prompts, report pass/fail."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def test(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append((name, condition, details))
        print(f"  [{status}] {name} {details}")

    def run(self):
        W = 70
        print("=" * W)
        print("  INTEGRATION BENCHMARK — END-TO-END PIPELINE")
        print("=" * W)

        # ── Setup pipeline once ─────────────────────────────────────────
        print("\n[SETUP] Loading pipeline ...")
        t0 = time.time()
        try:
            from src.pipeline import Pipeline, PipelineConfig
            cfg = PipelineConfig(
                use_t5_parser=False,       # rules parser (fast, no GPU for M1)
                use_motion_generation=True,
                use_asset_generation=False,
                use_ai_enhancement=False,
                duration=1.5,
                fps=12,
                output_dir="outputs/integration_test",
            )
            pipeline = Pipeline(cfg)
            pipeline.setup()
            setup_ok = True
        except Exception as exc:
            setup_ok = False
            print(f"  SETUP FAILED: {exc}")

        setup_time = time.time() - t0
        self.test("0. Pipeline setup", setup_ok, f"{setup_time:.1f}s")
        if not setup_ok:
            self._print_summary()
            return

        # ── Run prompts ────────────────────────────────────────────────
        print(f"\n[PROMPTS] Running {len(_PROMPTS)} diverse prompts ...")
        print("-" * 50)

        for i, (tag, prompt) in enumerate(_PROMPTS, start=1):
            self._run_one(pipeline, i, tag, prompt)

        # ── Cleanup check ──────────────────────────────────────────────
        print("\n[CLEANUP]")
        print("-" * 50)
        video_dir = os.path.join(cfg.output_dir, "videos")
        mp4s = [f for f in os.listdir(video_dir) if f.endswith(".mp4")] if os.path.isdir(video_dir) else []
        self.test(f"{len(_PROMPTS) + 1}. Video files on disk",
                  len(mp4s) >= len(_PROMPTS) * 0.8,
                  f"{len(mp4s)} files")

        self._print_summary()

    def _run_one(self, pipeline, idx: int, tag: str, prompt: str):
        """Run a single prompt through the pipeline and validate output."""
        t0 = time.time()
        try:
            result = pipeline.run(prompt, output_name=tag)
            elapsed = time.time() - t0
        except Exception as exc:
            self.test(f"{idx}. '{tag}'", False, f"ERROR: {exc}")
            return

        parsed = result.get("parsed_scene")
        frames = result.get("physics_frames", [])
        video  = result.get("video_path", "")

        has_entities = parsed is not None and len(parsed.entities) > 0
        has_frames   = len(frames) > 0
        has_video    = os.path.isfile(video) and os.path.getsize(video) > 1000

        ok = has_entities and has_frames and has_video
        details = (
            f"ent={len(parsed.entities) if parsed else 0} "
            f"act={len(parsed.actions) if parsed else 0} "
            f"frames={len(frames)} "
            f"{'video OK' if has_video else 'NO VIDEO'} "
            f"{elapsed:.1f}s"
        )
        self.test(f"{idx}. {tag}", ok, details)

    def _print_summary(self):
        total = self.passed + self.failed
        W = 70
        print()
        print("=" * W)
        print(f"INTEGRATION BENCHMARK: {self.passed}/{total} PASSED")
        print("=" * W)

        if self.failed > 0:
            print("\nFailed tests:")
            for name, ok, details in self.results:
                if not ok:
                    print(f"  - {name}: {details}")

        sys.exit(0 if self.failed == 0 else 1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    IntegrationBenchmark().run()

#!/usr/bin/env python3
"""
Run the full pipeline on 10 diverse showcase prompts at best quality.

Each prompt exercises different parts of the system:
  - M1 T5 parsing (entities, actions, relations)
  - M2 constraint-based layout
  - M4 SSM / semantic-retrieval motion
  - M5 PyBullet physics simulation
  - M7 cinematic post-processing (motion blur, DoF, color grade, vignette)

Usage:
    python scripts/run_showcase.py
    python scripts/run_showcase.py --duration 8 --fps 30
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import Pipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 10 showcase prompts — human actions & interactions ────────────────────────
PROMPTS = [
    # 1. Person walking — basic humanoid locomotion  
    ("01_person_walks", "a person walks forward slowly"),

    # 2. Person running — faster motion clip
    ("02_person_runs", "a man runs quickly across the room"),

    # 3. Person jumping — vertical motion
    ("03_person_jumps", "a person jumps over a small box"),

    # 4. Person kicks a ball — actor + object interaction
    ("04_kick_ball", "a person kicks a red ball on the ground"),

    # 5. Person throws an object — manipulation action
    ("05_throw_ball", "a man throws a blue ball forward"),

    # 6. Person pushes furniture — interaction with heavy object
    ("06_push_table", "a person pushes a brown table across the floor"),

    # 7. Person picks up a box — manipulation + spatial
    ("07_pick_up", "a woman picks up a green cube from the table"),

    # 8. Person waves — gesture action
    ("08_wave", "a person stands near a table and waves hello"),

    # 9. Multi-action sequence — walk + kick
    ("09_walk_kick", "a person walks toward a ball and kicks it"),

    # 10. Complex scene — person + multiple objects + action
    ("10_complex", "a man runs to a red ball near a blue cube and kicks it"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0,
                    help="Duration per video in seconds (default: 6)")
    ap.add_argument("--fps", type=int, default=24,
                    help="Frames per second (default: 24)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--outdir", default="outputs/showcase",
                    help="Output directory (default: outputs/showcase)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    config = PipelineConfig(
        output_dir=args.outdir,
        fps=args.fps,
        duration=args.duration,
        device=args.device,
        use_t5_parser=True,
        use_motion_generation=True,
        use_asset_generation=False,
        use_ai_enhancement=False,
        use_render_engine=True,
    )

    pipeline = Pipeline(config)
    pipeline.setup()

    results = []
    t_total = time.perf_counter()

    for idx, (name, prompt) in enumerate(PROMPTS, 1):
        sep = "=" * 60
        log.info("\n%s\n  [%d/10]  %s\n  Prompt: %s\n%s",
                 sep, idx, name, prompt, sep)

        t0 = time.perf_counter()
        try:
            result = pipeline.run(prompt, output_name=name)
            elapsed = time.perf_counter() - t0
            vpath = result.get("video_path", "?")
            nframes = len(result.get("physics_frames", []))

            log.info("  [OK] %s  (%d frames, %.1fs)", vpath, nframes, elapsed)
            results.append({
                "name": name, "prompt": prompt, "video": vpath,
                "frames": nframes, "time_s": round(elapsed, 1), "ok": True,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("  [FAIL] %s — %s (%.1fs)", name, exc, elapsed)
            results.append({
                "name": name, "prompt": prompt,
                "time_s": round(elapsed, 1), "ok": False, "error": str(exc),
            })

    total_time = time.perf_counter() - t_total

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SHOWCASE RESULTS")
    print("=" * 70)
    ok_count = sum(1 for r in results if r["ok"])
    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        frames = r.get("frames", 0)
        print(f"  [{status}]  {r['name']:25s}  {frames:4d} frames  {r['time_s']:6.1f}s")
    print("-" * 70)
    print(f"  {ok_count}/10 passed  |  total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Videos in: {os.path.abspath(args.outdir)}/videos/")
    print("=" * 70)

    # Save summary
    import json
    summary_path = os.path.join(args.outdir, "showcase_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()

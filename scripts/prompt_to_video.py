#!/usr/bin/env python3
"""End-to-end: text prompt → validated motion → SMPL silhouette video.

Chains the LangGraph motion pipeline (generate + validate + repair) with the
SMPL mesh renderer to produce a cinematic MP4 from a single natural-language
prompt.

Usage
-----
::

    python scripts/prompt_to_video.py "a person walks forward then kicks"
    python scripts/prompt_to_video.py "walk for 3 seconds, punch combo, victory" --out outputs/my_video.mp4
    python scripts/prompt_to_video.py "jump, spin, stand" --fps 30 --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.motion_pipeline import run_pipeline
from scripts.render_sequence import render_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger(__name__)


def prompt_to_video(
    prompt: str,
    out_video: str | None = None,
    fps: int = 24,
    max_repairs: int = 3,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Generate a motion video from a text prompt.

    Parameters
    ----------
    prompt : str
        Natural-language action description.
        Examples:
          - "a person walks forward"
          - "walk for 2 seconds, kick, punch combo, victory"
          - "jump then spin then stand"
    out_video : str or None
        Output MP4 path. Auto-generated from prompt if None.
    fps : int
        Frames per second.
    max_repairs : int
        Repair passes per segment in the validator loop.
    width, height : int
        Video resolution.

    Returns
    -------
    str — absolute path to the saved MP4.
    """
    t0 = time.time()

    # ── Derive file names ──
    safe_name = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)
    safe_name = safe_name.strip().replace(" ", "_")[:60] or "output"

    out_dir = "outputs/videos"
    os.makedirs(out_dir, exist_ok=True)

    npy_path = os.path.join(out_dir, f"{safe_name}.npy")
    if out_video is None:
        out_video = os.path.join(out_dir, f"{safe_name}.mp4")

    # ── Step 1: Generate validated motion via LangGraph pipeline ──
    log.info("=" * 70)
    log.info("  PROMPT → VIDEO PIPELINE")
    log.info("  Prompt: %r", prompt)
    log.info("=" * 70)
    log.info("")
    log.info("Step 1/2  Generating motion…")

    result = run_pipeline(
        description=prompt,
        fps=fps,
        max_repairs=max_repairs,
        output_path=npy_path,
    )

    report = result.get("final_report", "")
    log.info("  Motion: %s", report.split("\n")[0] if report else "done")

    # ── Step 2: Render to video ──
    log.info("")
    log.info("Step 2/2  Rendering video…")

    video_path = render_sequence(
        seq_path=npy_path,
        out_path=out_video,
        fps=fps,
        width=width,
        height=height,
    )

    elapsed = time.time() - t0
    abs_path = os.path.abspath(video_path)

    log.info("")
    log.info("=" * 70)
    log.info("  DONE  %.1f s total", elapsed)
    log.info("  Video: %s", abs_path)
    log.info("  Motion NPY: %s", os.path.abspath(npy_path))
    log.info("=" * 70)

    return abs_path


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Text prompt → validated motion → SMPL silhouette video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/prompt_to_video.py "a person walks forward"
  python scripts/prompt_to_video.py "walk for 3s, kick, victory" --out my.mp4
  python scripts/prompt_to_video.py "jump, spin, punch combo" --fps 30
""",
    )
    ap.add_argument("prompt", help="Natural-language action description")
    ap.add_argument("--out",    default=None, help="Output MP4 path")
    ap.add_argument("--fps",    type=int, default=24)
    ap.add_argument("--width",  type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-repairs", type=int, default=3)
    args = ap.parse_args()

    path = prompt_to_video(
        prompt=args.prompt,
        out_video=args.out,
        fps=args.fps,
        max_repairs=args.max_repairs,
        width=args.width,
        height=args.height,
    )
    print(f"\nVideo saved: {path}")

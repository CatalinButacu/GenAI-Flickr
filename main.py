#!/usr/bin/env python3
"""Physics-Constrained Video Generation — text prompt → MP4."""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import Pipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Physics-Constrained Video Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "a red ball falls onto a blue cube"\n'
            '  python main.py "a person kicks a ball" --duration 8 --fps 30\n'
        ),
    )
    p.add_argument("prompt", nargs="?", default="a red ball falls onto a blue cube")
    p.add_argument("--name", default="output")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--with-assets", action="store_true")
    p.add_argument("--with-enhance", action="store_true")
    p.add_argument("--motion", dest="motion", action="store_true", default=True)
    p.add_argument("--no-motion", dest="motion", action="store_false")
    return p.parse_args()


def main() -> None:
    args = _args()
    config = PipelineConfig(
        fps=args.fps,
        duration=args.duration,
        device=args.device,
        use_asset_generation=args.with_assets,
        use_motion_generation=args.motion,
        use_ai_enhancement=args.with_enhance,
    )
    result = Pipeline(config).run(args.prompt, output_name=args.name)
    video = result.get("video_path", "")
    parsed = result.get("parsed_scene")
    print(f"\nvideo  → {video}")
    if parsed:
        print(f"entities: {[e.name for e in parsed.entities]}")
        print(f"actions : {[a.action_type for a in parsed.actions]}")


if __name__ == "__main__":
    main()

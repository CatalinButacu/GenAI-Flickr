"""
Physics-constrained human motion demo
======================================
Runs the FULL pipeline for a text prompt:

  M1 → M2 → M4 → M5 → (M8) → M7
  NLP  scene  KIT-ML  PyBullet  ControlNet  cinematic
  parse plan  motion  physics   human       post-proc

Two rendering paths:

  **Default (skeleton):**  Physics-verified glow skeleton (fast, no GPU needed)
  **--controlnet:**        SD 1.5 + ControlNet OpenPose → photorealistic human
                           (~10-15 s/frame on RTX 3050, downloads ~5 GB on first run)

Physics constraints (in both paths):
  - Gravity on every rigid body link
  - Ground-contact forces and friction
  - Joint force limits from humanoid.urdf
  - RSI root tracking of KIT-ML pelvis trajectory
  - PD motors for retargeted joint angles

Usage:
    python examples/demo_physics_human.py
    python examples/demo_physics_human.py --controlnet
    python examples/demo_physics_human.py --controlnet --duration 3 --fps 12
    python examples/demo_physics_human.py --prompt "a person runs forward"
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SEP = "━" * 51

_SEP = "\u2501" * 51


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-constrained human motion demo")
    p.add_argument("--prompt",    default="a person walks forward",
                   help="Natural-language prompt for the scene")
    p.add_argument("--name",      default="physics_human",
                   help="Output filename stem")
    p.add_argument("--duration",  type=float, default=5.0,
                   help="Simulation duration in seconds")
    p.add_argument("--fps",       type=int,   default=24,
                   help="Output video frames-per-second")
    p.add_argument("--device",    default="cpu",
                   help="Torch device (cpu or cuda — not needed for base pipeline)")
    p.add_argument("--no-render-engine", action="store_true",
                   help="Skip M7 post-processing")
    p.add_argument("--controlnet", action="store_true",
                   help="Use ControlNet OpenPose for photorealistic human "
                        "(~10-15 s/frame, downloads ~5 GB on first run)")
    p.add_argument("--animatediff", action="store_true",
                   help="Use AnimateDiff + ControlNet for temporally-consistent "
                        "photorealistic human video (downloads ~1.8 GB on first run)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── configure and build the pipeline ─────────────────────────────────────
    from src.pipeline import Pipeline, PipelineConfig

    # When using ControlNet, lower fps (slow inference) and force CUDA
    fps = args.fps
    device = args.device
    use_ai = args.controlnet or args.animatediff
    if use_ai:
        fps = min(fps, 12)          # keep frame count manageable
        device = "cuda"             # AI rendering requires GPU
        mode = "AnimateDiff" if args.animatediff else "ControlNet"
        log.info("%s mode — fps capped at %d, device=%s", mode, fps, device)

    config = PipelineConfig(
        output_dir="outputs",
        fps=fps,
        duration=args.duration,
        device=device,
        use_asset_generation=False,   # M3: Shap-E  — not needed here
        use_motion_generation=True,   # M4: KIT-ML retrieval + SSM
        use_ai_enhancement=use_ai,    # M8: AnimateDiff or ControlNet
        use_render_engine=not args.no_render_engine,  # M7: cinematic post-proc
    )

    pipeline = Pipeline(config)

    # ── run the full pipeline ─────────────────────────────────────────────────
    log.info(_SEP)
    log.info("Prompt   : %s", args.prompt)
    log.info("Duration : %.1f s  |  FPS : %d", args.duration, fps)
    log.info(_SEP)

    result = pipeline.run(args.prompt, output_name=args.name)

    # ── report ────────────────────────────────────────────────────────────────
    video_path = result.get("video_path", "")
    parsed     = result.get("parsed_scene")
    clips      = result.get("motion_clips", {})

    log.info(_SEP)
    log.info("PIPELINE COMPLETE")
    log.info("")
    if parsed:
        log.info("  M1 entities : %s", [e.name for e in parsed.entities])
        log.info("  M1 actions  : %s", [a.action_type for a in parsed.actions])
    if clips:
        for actor, clip in clips.items():
            if clip is not None:
                has_raw = clip.raw_joints is not None
                log.info("  M4 clip     : actor='%s'  action='%s'  frames=%d  "
                         "raw_joints=%s  source=%s",
                         actor, clip.action, clip.num_frames, has_raw, clip.source)
    log.info("  M5 frames   : %d", len(result.get("physics_frames", [])))
    log.info("  Video       : %s", video_path)
    log.info(_SEP)

    # ── open the video ────────────────────────────────────────────────────────
    if os.path.exists(video_path):
        abs_path = os.path.abspath(video_path)
        subprocess.Popen(["cmd", "/c", "start", "", abs_path])
        print(f"\nVideo saved → {abs_path}")
    else:
        print("\n[!] Video was not created — check log for errors.")


if __name__ == "__main__":
    main()

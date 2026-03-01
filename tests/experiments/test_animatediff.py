"""
AnimateDiff + ControlNet test  —  8-frame temporally-consistent clip.

Downloads the AnimateDiff motion adapter (~1.8 GB) on first run.
Generates 8 photorealistic frames WITH cross-frame temporal attention,
so identity/clothing/skin stay consistent between frames.

Run:   python examples/test_animatediff.py
Time:  ~5-10 min on RTX 3050 (download + inference)
VRAM:  ~4.0-4.5 GB (borderline on 4GB, uses CPU offload)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    import torch

    out_dir = "outputs/animatediff_test"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load walking clip ───────────────────────────────────────────────
    log.info("Loading KIT-ML motion…")
    from src.modules.motion_generator import MotionGenerator

    mg = MotionGenerator(use_retrieval=True, use_ssm=False)
    clip = mg.generate("walk forward", num_frames=60)
    if clip is None or clip.raw_joints is None:
        log.error("No raw_joints")
        return

    raw = clip.raw_joints
    log.info("Clip: %s, %d frames", clip.action, len(raw))

    # ── Pick 4 evenly-spaced frames (4 fits in 4GB VRAM) ───────────────
    n_frames = 4
    indices = np.linspace(0, len(raw) - 1, n_frames, dtype=int)
    selected = [raw[i] for i in indices]

    # ── Project to OpenPose (384×384 to reduce VRAM) ────────────────────
    from src.modules.ai_enhancer import SkeletonProjector

    projector = SkeletonProjector(img_w=384, img_h=384, cam_yaw_deg=15.0)
    skeleton_imgs = [projector.render(xyz) for xyz in selected]

    for i, (skel, idx) in enumerate(zip(skeleton_imgs, indices)):
        path = os.path.join(out_dir, f"skeleton_{i:02d}_f{idx:03d}.png")
        cv2.imwrite(path, cv2.cvtColor(skel, cv2.COLOR_RGB2BGR))
    log.info("Saved %d skeleton images", n_frames)

    # ── Load AnimateDiff + ControlNet ───────────────────────────────────
    log.info("Loading AnimateDiff pipeline (downloads ~1.8 GB on first run)…")
    from src.modules.ai_enhancer import AnimateDiffHumanRenderer

    renderer = AnimateDiffHumanRenderer(
        device="cuda",
        num_steps=15,
        seed=42,
        batch_size=n_frames,
        width=384,
        height=384,
    )

    t_setup = time.time()
    ok = renderer.setup()
    if not ok:
        log.error("AnimateDiff setup failed — check error above")
        log.info("Falling back to per-frame ControlNet for comparison…")

        from src.modules.ai_enhancer import ControlNetHumanRenderer
        cn = ControlNetHumanRenderer(device="cuda", num_steps=15)
        cn.setup()

        log.info("Generating %d per-frame ControlNet frames (same seed)…", n_frames)
        t0 = time.time()
        frames = cn.render_sequence(skeleton_imgs)
        elapsed = time.time() - t0
        log.info("Per-frame ControlNet: %d frames in %.0fs", len(frames), elapsed)

        _save_results(out_dir, skeleton_imgs, frames, indices, elapsed)
        del cn
        torch.cuda.empty_cache()
        return

    setup_time = time.time() - t_setup
    log.info("AnimateDiff loaded in %.0fs", setup_time)

    # ── Generate 8 temporally-consistent frames ─────────────────────────
    log.info("Generating %d frames with AnimateDiff…", n_frames)
    t0 = time.time()
    frames = renderer.render_batch(skeleton_imgs)
    elapsed = time.time() - t0
    log.info("AnimateDiff: %d frames in %.0fs (%.1fs/frame)",
             len(frames), elapsed, elapsed / len(frames))

    _save_results(out_dir, skeleton_imgs, frames, indices, elapsed)

    # ── Clean up ────────────────────────────────────────────────────────
    del renderer
    torch.cuda.empty_cache()


def _save_results(
    out_dir: str,
    skeleton_imgs: list,
    frames: list,
    indices: np.ndarray,
    elapsed: float,
) -> None:
    """Save individual frames, grid, and video."""
    import subprocess

    n_frames = len(frames)

    # Save individual frames
    for i, frame in enumerate(frames):
        path = os.path.join(out_dir, f"photorealistic_{i:02d}_f{indices[i]:03d}.png")
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    log.info("Saved %d photorealistic frames", n_frames)

    # Create comparison grid: top = skeleton, bottom = photorealistic
    h, w = 384, 384
    cols = min(4, n_frames)
    rows_per_set = (n_frames + cols - 1) // cols
    grid_h = 2 * rows_per_set * h
    grid_w = cols * w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for i in range(n_frames):
        r = i // cols
        c = i % cols
        # Skeleton row
        skel_r = cv2.resize(skeleton_imgs[i], (w, h))
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = skel_r
        # Photorealistic row (offset by rows_per_set)
        photo_r = cv2.resize(frames[i], (w, h))
        grid[(rows_per_set+r)*h:(rows_per_set+r+1)*h, c*w:(c+1)*w] = photo_r

    cv2.putText(grid, "SKELETON INPUT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(grid, "AI RENDER (AnimateDiff+ControlNet)", (10, rows_per_set*h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)

    grid_path = os.path.join(out_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    log.info("Comparison grid → %s", grid_path)

    # Create short video from the photorealistic frames
    video_path = os.path.join(out_dir, "animatediff_clip.mp4")
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(video_path, fps=8, codec='libx264',
                                    quality=8, pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        log.info("Video → %s", video_path)
    except Exception as exc:
        log.warning("Video export failed: %s", exc)

    # Open results
    subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(grid_path)])

    print(f"\n{'━' * 55}")
    print(f"  Frames : {n_frames}")
    print(f"  Time   : {elapsed:.0f}s ({elapsed/n_frames:.1f}s/frame)")
    print(f"  Grid   : {os.path.abspath(grid_path)}")
    print(f"  Video  : {os.path.abspath(video_path)}")
    print(f"{'━' * 55}")


if __name__ == "__main__":
    main()

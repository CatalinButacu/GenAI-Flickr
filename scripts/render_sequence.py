#!/usr/bin/env python3
"""Render a (N, 21, 3) joint-position sequence to MP4.

Uses the same SilhouetteSkeletonRenderer as the cinematic demo so the
output looks identical (SMPL mesh silhouette + ground grid + HUD).

Usage
-----
    python scripts/render_sequence.py                        # default: outputs/long_sequence.npy
    python scripts/render_sequence.py outputs/my_seq.npy    # custom file
    python scripts/render_sequence.py --fps 24 --width 1280 --height 720
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.physics.body_model import BodyParams
from src.modules.physics.silhouette_renderer import SilhouetteSkeletonRenderer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Camera path: slow 180° orbit, tracking the pelvis
# ALL camera coordinates are in METRES (the renderer divides xyz_mm/1000
# internally, but the *target* is fed directly to the camera matrix).
# ──────────────────────────────────────────────────────────────────────

def _camera_at(t: float, pelvis_mm: np.ndarray) -> dict:
    """Return camera parameters for normalised time t ∈ [0, 1].

    target tracks the pelvis in world metres so the body stays centred.
    """
    yaw   = 30.0 + 180.0 * t
    pitch = -18.0 + math.sin(t * math.pi) * (-8.0)   # slight dip at peak
    dist  = 3.8 - 0.4 * math.sin(t * math.pi)         # push in at mid
    # Body centre is about 0.3 m above the pelvis (mid-torso)
    target_x =  pelvis_mm[0] / 1000.0   # mm → m
    target_y =  pelvis_mm[1] / 1000.0 + 0.30
    target_z =  pelvis_mm[2] / 1000.0
    return dict(yaw=yaw, pitch=pitch, distance=dist,
                target=[target_x, target_y, target_z])


# ──────────────────────────────────────────────────────────────────────
# HUD overlay
# ──────────────────────────────────────────────────────────────────────

def _draw_hud(
    frame: np.ndarray,
    idx: int,
    n: int,
    fps: int,
    action_name: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Bottom bar
    cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Frame / time counter
    sec = idx / fps
    total_sec = n / fps
    cv2.putText(frame,
                f"{action_name}    {sec:.1f}s / {total_sec:.1f}s    frame {idx+1}/{n}",
                (18, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (210, 200, 185), 1, cv2.LINE_AA)

    # Progress bar
    bar_x = int(w * idx / max(n - 1, 1))
    cv2.rectangle(frame, (0, h - 4), (bar_x, h), (90, 180, 255), -1)
    return frame


# ──────────────────────────────────────────────────────────────────────
# Main renderer
# ──────────────────────────────────────────────────────────────────────

def render_sequence(
    seq_path: str,
    out_path: str | None = None,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> str:
    seq = np.load(seq_path)
    if seq.ndim != 3 or seq.shape[1:] != (21, 3):
        raise ValueError(f"Expected shape (N, 21, 3), got {seq.shape}")

    n = seq.shape[0]
    if out_path is None:
        base = os.path.splitext(seq_path)[0]
        out_path = base + ".mp4"

    action_name = os.path.splitext(os.path.basename(seq_path))[0].replace("_", " ")
    log.info("Rendering %d frames → %s", n, out_path)
    # Root-normalise: keep the character in-place (pelvis at X=0, Z=0).
    # Each joint is translated by the pelvis offset so the whole body
    # stays centred in the frame regardless of generator root translation.
    seq = seq.copy()
    seq[:, :, 0] -= seq[:, 0:1, 0]   # subtract pelvis X per frame
    seq[:, :, 2] -= seq[:, 0:1, 2]   # subtract pelvis Z per frame
    body = BodyParams(height_m=1.78, muscle_mass=1.0, body_fat=1.0, gender="neutral")

    cam0 = _camera_at(0.0, seq[0, 0])
    renderer = SilhouetteSkeletonRenderer(
        img_w=width, img_h=height,
        yaw_deg=cam0["yaw"],
        pitch_deg=cam0["pitch"],
        distance=cam0["distance"],
        target=cam0["target"],
        action_label=action_name,
        body_params=body,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p")

    for i, pose in enumerate(seq):
        t = i / max(n - 1, 1)
        cam = _camera_at(t, pose[0])   # pass current pelvis position
        renderer.update_camera(cam["yaw"], cam["pitch"], cam["distance"], cam["target"])

        prev = seq[i - 1] if i > 0 else None
        bgr = renderer.render_frame(pose, prev_xyz=prev)

        bgr = _draw_hud(bgr, i, n, fps, action_name)
        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if (i + 1) % 48 == 0 or i == n - 1:
            log.info("  %d / %d  (%.0f%%)", i + 1, n, 100.0 * (i + 1) / n)

    writer.close()
    abs_path = os.path.abspath(out_path)
    log.info("Saved → %s", abs_path)
    return abs_path


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render (N,21,3) NPY sequence to MP4")
    parser.add_argument("seq", nargs="?", default="outputs/long_sequence.npy",
                        help="Path to (N, 21, 3) .npy file")
    parser.add_argument("--out", default=None, help="Output MP4 path")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    path = render_sequence(
        args.seq,
        out_path=args.out,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    print(f"\nVideo saved: {path}")

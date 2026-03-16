"""
Headless aitviewer demo — renders an Inter-X two-person SMPL-X sequence to video.

Reads real motion data from data/inter-x/motions/<sample>/P1.npz + P2.npz,
builds SMPLSequence renderables for both persons, and exports a .mp4 video
using aitviewer's HeadlessRenderer — no interactive window required.

Usage:
    python scripts/demo_interx_viewer.py
    python scripts/demo_interx_viewer.py --sample G001T000A000R002 --fps 30
    python scripts/demo_interx_viewer.py --out outputs/interx_demo.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── project root on sys.path ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── SMPL-X body models: use arctic unpack models ──────────────────────────────
_SMPLX_MODELS = os.path.join(ROOT, "data", "arctic", "unpack", "models")

from aitviewer.configuration import CONFIG as C

C.update_conf({"smplx_models": _SMPLX_MODELS, "window_width": 1280, "window_height": 720})

from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.plane import Plane


# Person colours (blue / orange)
COLOR_P1 = (0.11, 0.53, 0.80, 1.0)
COLOR_P2 = (1.00, 0.27, 0.00, 1.0)


def load_smplx_sequence(npz_path: str, color: tuple) -> SMPLSequence:
    """Load a P1.npz / P2.npz file and return a SMPLSequence renderable."""
    data = np.load(npz_path, allow_pickle=True)

    T = data["pose_body"].shape[0]
    gender = str(data["gender"])

    poses_body  = data["pose_body"].reshape(T, -1)    # (T, 63)
    poses_lhand = data["pose_lhand"].reshape(T, -1)   # (T, 45)
    poses_rhand = data["pose_rhand"].reshape(T, -1)   # (T, 45)
    poses_root  = data["root_orient"]                  # (T, 3)
    betas       = data["betas"]                        # (1, 10)
    trans       = data["trans"]                        # (T, 3)

    smpl_layer = SMPLLayer(
        model_type="smplx", gender=gender, num_betas=10, device=C.device
    )
    seq = SMPLSequence(
        poses_body=poses_body,
        smpl_layer=smpl_layer,
        poses_root=poses_root,
        betas=betas,
        trans=trans,
        poses_left_hand=poses_lhand,
        poses_right_hand=poses_rhand,
        device=C.device,
        color=color,
    )
    return seq


def build_renderer(seq_p1: SMPLSequence, seq_p2: SMPLSequence) -> HeadlessRenderer:
    """Create a HeadlessRenderer with both persons and a ground plane."""
    renderer = HeadlessRenderer()

    # Ground plane
    ground = Plane(
        center=np.array([0.0, 0.0, 0.0]),
        size=8.0,
        color=(0.9, 0.9, 0.9, 1.0),
    )
    renderer.scene.add(ground)
    renderer.scene.add(seq_p1)
    renderer.scene.add(seq_p2)

    # Orient camera for a nice two-person view
    renderer.scene.camera.position = np.array([0.0, 1.5, 4.5])
    renderer.scene.camera.target = np.array([0.0, 0.9, 0.0])

    return renderer


def main():
    parser = argparse.ArgumentParser(description="Headless Inter-X SMPL-X viewer demo")
    parser.add_argument(
        "--sample",
        default="G001T000A000R000",
        help="Inter-X sample directory name under data/inter-x/motions/",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "outputs", "interx_demo.mp4"),
        help="Output video path (.mp4)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    args = parser.parse_args()

    motions_dir = os.path.join(ROOT, "data", "inter-x", "motions")
    sample_dir = os.path.join(motions_dir, args.sample)

    if not os.path.isdir(sample_dir):
        # Pick first available sample if the default doesn't exist
        available = sorted(os.listdir(motions_dir))
        if not available:
            print(f"ERROR: No samples found in {motions_dir}")
            sys.exit(1)
        args.sample = available[0]
        sample_dir = os.path.join(motions_dir, args.sample)
        print(f"[warn] Requested sample not found; using {args.sample}")

    p1_path = os.path.join(sample_dir, "P1.npz")
    p2_path = os.path.join(sample_dir, "P2.npz")
    print(f"Loading sample: {args.sample}")
    print(f"  P1: {p1_path}")
    print(f"  P2: {p2_path}")

    seq_p1 = load_smplx_sequence(p1_path, COLOR_P1)
    seq_p2 = load_smplx_sequence(p2_path, COLOR_P2)

    n_frames = min(seq_p1.n_frames, seq_p2.n_frames)
    print(f"  frames P1={seq_p1.n_frames}  P2={seq_p2.n_frames}  → rendering {n_frames}")

    renderer = build_renderer(seq_p1, seq_p2)
    renderer.scene.fps = args.fps

    # Try to load text description
    texts_dir = os.path.join(ROOT, "data", "inter-x", "texts")
    txt_path = os.path.join(texts_dir, args.sample + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path) as f:
            desc = f.read().strip()
        print(f"  text: {desc[:120]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Use a temp frames dir alongside the output
    frames_dir = os.path.join(os.path.dirname(args.out), "_frames_tmp")
    print(f"\nRendering {n_frames} frames → {args.out} ...")
    renderer.save_video(
        frame_dir=frames_dir,
        video_dir=args.out,
        output_fps=args.fps,
    )
    print(f"Done → {args.out}")


if __name__ == "__main__":
    main()

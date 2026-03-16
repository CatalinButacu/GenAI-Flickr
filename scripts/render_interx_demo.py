"""Headless aitviewer demo — render Inter-X two-person SMPL-X interaction.

Usage:
    python scripts/render_interx_demo.py
    python scripts/render_interx_demo.py --sample G001T000A000R000 --fps 30 --out outputs/interx_demo.mp4

What it does:
  1. Loads a two-person Inter-X sample (P1.npz + P2.npz)
  2. Builds two coloured SMPLSequence renderables (blue + orange)
  3. Adds a ground-plane checkerboard
  4. Renders every frame headlessly via aitviewer's HeadlessRenderer
  5. Assembles the PNG frames into an MP4 with OpenCV
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import tempfile

import numpy as np

# ─── Paths ------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_MOTIONS_DIR = os.path.join(_ROOT, "data", "inter-x", "motions")
_TEXTS_DIR   = os.path.join(_ROOT, "data", "inter-x", "texts")
_SMPLX_DIR   = os.path.join(_ROOT, "data", "arctic", "unpack", "models")
_OUT_DIR     = os.path.join(_ROOT, "outputs", "interx_demo")

# Configure aitviewer BEFORE any other aitviewer import
from aitviewer.configuration import CONFIG as C
C.update_conf({
    "smplx_models": _SMPLX_DIR,
    "window_width":  1280,
    "window_height": 720,
})

from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer


# ─── Helpers ----------------------------------------------------------------

def load_smplx_sequence(npz_path: str, color: tuple, max_frames: int = 0) -> SMPLSequence:
    """Load a SMPL-X .npz file and return an aitviewer SMPLSequence."""
    data = np.load(npz_path, allow_pickle=True)
    T = data["pose_body"].shape[0]
    if max_frames > 0:
        T = min(T, max_frames)

    gender = str(data["gender"])
    betas        = data["betas"]                                  # (10,)
    root_orient  = data["root_orient"][:T]                        # (T, 3)
    pose_body    = data["pose_body"][:T].reshape(T, -1)           # (T, 63)
    pose_lhand   = data["pose_lhand"][:T].reshape(T, -1)          # (T, 45)
    pose_rhand   = data["pose_rhand"][:T].reshape(T, -1)          # (T, 45)
    trans        = data["trans"][:T]                              # (T, 3)

    smpl_layer = SMPLLayer(
        model_type="smplx",
        gender=gender,
        num_betas=min(10, len(betas)),
        device=C.device,
    )
    seq = SMPLSequence(
        poses_body=pose_body,
        smpl_layer=smpl_layer,
        poses_root=root_orient,
        betas=betas,
        trans=trans,
        poses_left_hand=pose_lhand,
        poses_right_hand=pose_rhand,
        device=C.device,
        color=color,
    )
    return seq


def read_text(sample_id: str) -> str:
    txt_path = os.path.join(_TEXTS_DIR, sample_id + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "(no text annotation)"


def assemble_video(frame_dir: str, out_path: str, fps: int) -> None:
    """Collect PNG frames produced by aitviewer and write an MP4.

    aitviewer saves frames under <frame_dir>/<scene_name>/  (e.g. .../0000/)
    so we walk into the first subdirectory if needed.
    """
    import cv2

    # aitviewer puts frames one level deeper in a scene-named subfolder
    search_dir = frame_dir
    subdirs = [os.path.join(frame_dir, d) for d in os.listdir(frame_dir)
               if os.path.isdir(os.path.join(frame_dir, d))]
    if subdirs:
        search_dir = subdirs[0]   # e.g. .../0000/

    frames = sorted(
        f for f in os.listdir(search_dir)
        if f.lower().endswith(".png")
    )
    if not frames:
        print(f"[WARN] No PNG frames found in {search_dir}")
        return

    sample_img = cv2.imread(os.path.join(search_dir, frames[0]))
    h, w = sample_img.shape[:2]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for fname in frames:
        img = cv2.imread(os.path.join(search_dir, fname))
        writer.write(img)
    writer.release()
    print(f"[OK] Video saved -> {out_path}  ({len(frames)} frames @ {fps} fps)")


# ─── Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Headless Inter-X SMPL-X demo render")
    parser.add_argument("--sample", default="G001T000A000R000",
                        help="Sample ID from data/inter-x/motions/")
    parser.add_argument("--fps", type=int, default=30,
                        help="Playback / output FPS")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Limit rendered frames (0 = all)")
    parser.add_argument("--out", default="",
                        help="Output MP4 path (default: outputs/interx_demo/<sample>.mp4)")
    args = parser.parse_args()

    sample_dir = os.path.join(_MOTIONS_DIR, args.sample)
    if not os.path.isdir(sample_dir):
        sys.exit(f"[ERROR] Sample not found: {sample_dir}")

    p1_path = os.path.join(sample_dir, "P1.npz")
    p2_path = os.path.join(sample_dir, "P2.npz")

    out_path = args.out or os.path.join(_OUT_DIR, f"{args.sample}.mp4")

    text_desc = read_text(args.sample)
    print(f"[INFO] Sample : {args.sample}")
    print(f"[INFO] Text   : {text_desc}")
    print(f"[INFO] Output : {out_path}")

    # ── Load sequences ──────────────────────────────────────────────────────
    print("[INFO] Loading SMPL-X sequences …")
    seq_p1 = load_smplx_sequence(p1_path, color=(0.11, 0.53, 0.8, 1.0),
                                 max_frames=args.max_frames)  # blue
    seq_p2 = load_smplx_sequence(p2_path, color=(1.0,  0.27, 0.0, 1.0),
                                 max_frames=args.max_frames)  # orange

    n_frames = min(seq_p1.n_frames, seq_p2.n_frames)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    print(f"[INFO] Frames : {n_frames} @ {args.fps} fps  "
          f"({n_frames / args.fps:.1f} s)")

    # ── Build scene ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="aitviewer_frames_") as frame_dir:
        renderer = HeadlessRenderer()
        renderer.scene.fps = args.fps
        renderer.playback_fps = args.fps

        # Neutral studio background
        renderer.scene.background_color = [0.85, 0.87, 0.90, 1.0]

        # Replace checkerboard floor with near-solid neutral ground
        if renderer.scene.floor is not None:
            renderer.scene.remove(renderer.scene.floor)
        from aitviewer.renderables.plane import ChessboardPlane
        renderer.scene.floor = ChessboardPlane(
            100.0, 200,
            (0.82, 0.83, 0.84, 1.0),
            (0.80, 0.81, 0.82, 1.0),
            "xz", name="Floor",
        )
        renderer.scene.floor.material.diffuse = 0.1
        renderer.scene.add(renderer.scene.floor)

        # Two-person sequences
        renderer.scene.add(seq_p1)
        renderer.scene.add(seq_p2)


        # Position camera for a good two-person overview
        renderer.scene.camera.position = np.array([0.0, 1.5, 4.5])
        renderer.scene.camera.target   = np.array([0.0, 1.0, 0.0])

        print("[INFO] Rendering frames (this may take a minute) …")
        renderer.save_video(
            frame_dir=frame_dir,
            video_dir=None,   # we'll assemble manually for reliability
            output_fps=args.fps,
        )

        assemble_video(frame_dir, out_path, args.fps)


if __name__ == "__main__":
    main()

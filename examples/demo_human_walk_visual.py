"""Demo: KIT-ML walk motion → cinematic skeleton visualization (no model downloads).

Produces a broadcast-quality motion capture visualization with:
  • Radial gradient background  (dark studio look)
  • Glow / bloom on each bone   (multi-layer gaussian blur composite)
  • Depth shading               (near bones brighter + thicker)
  • Velocity colouring          (fast joints glow hot)
  • Ground shadow / floor grid
  • M7 post-processing          (motion blur, colour grade, vignette)

Skeleton preview is exported in under 5 seconds.
ControlNet path (photorealistic) is in demo_human_walk.py — run it when
models are downloaded (~5.5 GB, can run overnight alongside training).

Run:
    python examples/demo_human_walk_visual.py
    python examples/demo_human_walk_visual.py --yaw 20 --clips 3
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── KIT-ML skeleton connectivity (same as demo_human_walk.py) ─────────────
_BONES = [
    (0,1),(1,2),(2,3),(3,4),            # spine + head
    (2,5),(5,6),(6,7),                  # left arm
    (2,8),(8,9),(9,10),                 # right arm
    (0,11),(0,16),(11,16),              # pelvis → hips
    (11,12),(12,13),(13,14),(14,15),    # left leg
    (16,17),(17,18),(18,19),(19,20),    # right leg
    (5,8),                              # shoulder girdle
]

# HSV-style hue per bone (0-179 in OpenCV), saturation, value multiplier
_BONE_HSV = [
    (30,  180, 1.0),   # spine — amber
    (30,  180, 1.0),
    (30,  180, 1.0),
    (30,  180, 1.0),
    (100, 220, 1.0),   # L arm — cyan-blue
    (100, 220, 1.0),
    (100, 220, 1.0),
    (10,  240, 1.0),   # R arm — orange-red
    (10,  240, 1.0),
    (10,  240, 1.0),
    (60,  160, 0.75),  # pelvis — yellow-green
    (60,  160, 0.75),
    (60,  160, 0.75),
    (140, 200, 1.0),   # L leg — green-teal
    (140, 200, 1.0),
    (140, 200, 1.0),
    (140, 200, 1.0),
    (130, 200, 1.0),   # R leg — purple
    (130, 200, 1.0),
    (130, 200, 1.0),
    (130, 200, 1.0),
    (0,   200, 0.9),   # girdle — red
]


def _hsv_to_rgb(h: int, s: int, v_frac: float) -> tuple[int, int, int]:
    """OpenCV HSV (h 0-179) → RGB tuple."""
    import cv2
    import numpy as np
    px = np.array([[[h, s, int(255 * v_frac)]]], dtype=np.uint8)
    rgb = cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (same helper as demo_human_walk.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_clip(kit_dir: str, sample_id: str) -> tuple[np.ndarray, str]:
    kit = Path(kit_dir)
    jpath = kit / "new_joints" / f"{sample_id}.npy"
    tpath = kit / "texts"      / f"{sample_id}.txt"
    joints = np.load(jpath)
    text   = tpath.read_text().split("#")[0].strip() if tpath.exists() else sample_id
    return joints, text


def find_clips(kit_dir: str, query: str, n: int = 3) -> list[tuple[str, str, np.ndarray]]:
    """Return up to *n* clips whose text contains *query*."""
    kit        = Path(kit_dir)
    texts_dir  = kit / "texts"
    joints_dir = kit / "new_joints"
    results    = []
    for tpath in sorted(texts_dir.glob("*.txt")):
        raw = tpath.read_text().split("#")[0].strip().lower()
        if query in raw:
            jpath = joints_dir / (tpath.stem + ".npy")
            if jpath.exists():
                j = np.load(jpath)
                if 24 <= len(j) <= 120:
                    results.append((tpath.stem, raw, j))
                    if len(results) >= n:
                        break
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3-D → 2-D projection
# ─────────────────────────────────────────────────────────────────────────────

def project(joints_3d: np.ndarray, w: int, h: int, yaw: float) -> np.ndarray:
    """Orthographic projection with yaw rotation. Returns (21, 3) — (col, row, depth_norm)."""
    pts = joints_3d.copy()
    pts[:, 0] -= pts[0, 0]
    pts[:, 2] -= pts[0, 2]

    rad = np.radians(yaw)
    c, s = np.cos(rad), np.sin(rad)
    x_rot =  c * pts[:, 0] + s * pts[:, 2]
    z_rot = -s * pts[:, 0] + c * pts[:, 2]
    pts[:, 0] = x_rot
    pts[:, 2] = z_rot

    sx = pts[:, 0]
    sy = pts[:, 1]   # Y-up

    pad = 0.12
    x_r = sx.max() - sx.min() or 1.0
    y_r = sy.max() - sy.min() or 1.0
    scale = min(w * (1 - 2*pad) / x_r, h * (1 - 2*pad) / y_r)

    col  = (sx - sx.min()) * scale + w * pad
    row  = h - ((sy - sy.min()) * scale + h * pad)

    # Normalise depth 0..1 for brightness modulation
    z    = pts[:, 2]
    depth = (z - z.min()) / (z.max() - z.min() + 1e-6)

    return np.stack([col, row, depth], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Cinematic frame renderer
# ─────────────────────────────────────────────────────────────────────────────

def make_background(w: int, h: int) -> np.ndarray:
    """Dark radial gradient background with a subtle horizon glow."""
    import cv2
    import numpy as np

    # Radial vignette — dark corners, slightly lighter centre
    cx, cy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(cx**2 + cy**2)
    bg_v = np.clip(1 - 0.7 * r, 0.04, 0.28)

    # Ground plane glow — subtle warm band at the bottom third
    horizon_y = int(h * 0.72)
    glow = np.zeros((h, w), np.float32)
    glow[horizon_y:, :] = np.linspace(0, 1, h - horizon_y)[:, None] * 0.18
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=w * 0.04)

    r_ch = np.clip(bg_v + glow * 0.3, 0, 1)
    g_ch = np.clip(bg_v + glow * 0.2, 0, 1)
    b_ch = np.clip(bg_v + glow * 0.05, 0, 1)

    bg = np.stack([r_ch, g_ch, b_ch], axis=-1)
    return (bg * 255).astype(np.uint8)


def _draw_glow_line(canvas: np.ndarray, glow_layer: np.ndarray,
                    pt1: tuple, pt2: tuple,
                    color: tuple[int,int,int],
                    thickness: int) -> None:
    """Draw a bone with a soft glow: sharp core + blurred halo on glow_layer."""
    import cv2
    import numpy as np

    # Core line
    cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Glow: draw thick white line on glow_layer which gets blurred later
    intensity = int(sum(color) / 3)
    glow_color = (intensity, intensity, intensity)
    cv2.line(glow_layer, pt1, pt2, glow_color, thickness + 6, cv2.LINE_AA)


def render_cinematic_frame(uvd: np.ndarray,
                            joint_vel: np.ndarray,
                            bg: np.ndarray,
                            w: int, h: int) -> np.ndarray:
    """
    Render one cinematic motion-capture frame.

    uvd  — (21, 3) array: (col, row, depth_norm)
    joint_vel — (21,) per-joint speed norm for heat colouring
    bg   — pre-built background (H, W, 3)
    """
    import cv2
    import numpy as np

    canvas    = bg.copy().astype(np.float32) / 255.0
    glow_buf  = np.zeros((h, w, 3), np.float32)
    core_buf  = np.zeros((h, w, 3), np.float32)

    # ── draw bones ──────────────────────────────────────────────────────────
    for bone_idx, (ja, jb) in enumerate(_BONES):
        hue, sat, val_m = _BONE_HSV[bone_idx]

        depth_mid = (uvd[ja, 2] + uvd[jb, 2]) * 0.5
        vel_boost = float(np.clip((joint_vel[ja] + joint_vel[jb]) * 0.5 / 8.0, 0, 1))

        # Brighter / saturated for fast joints & near-camera joints
        v = float(np.clip(val_m * (0.55 + 0.45 * depth_mid + 0.25 * vel_boost), 0, 1))
        s = int(np.clip(sat * (0.7 + 0.3 * vel_boost), 0, 255))
        r, g, b = _hsv_to_rgb(hue, s, v)

        pt_a = (int(np.clip(uvd[ja, 0], 0, w-1)), int(np.clip(uvd[ja, 1], 0, h-1)))
        pt_b = (int(np.clip(uvd[jb, 0], 0, w-1)), int(np.clip(uvd[jb, 1], 0, h-1)))

        # Thickness scales with depth and velocity
        thick = max(2, int(2 + 4 * depth_mid + 2 * vel_boost))

        # Glow buf (will be blurred)
        gv = min(1.0, v + 0.2 + vel_boost * 0.3)
        cv2.line(glow_buf, pt_a, pt_b, (b * gv, g * gv, r * gv), thick + 10, cv2.LINE_AA)

        # Core buf (stay sharp)
        cv2.line(core_buf, pt_a, pt_b, (b, g, r), thick, cv2.LINE_AA)

    # ── blur glow and composite ──────────────────────────────────────────────
    glow_blurred = cv2.GaussianBlur(glow_buf / 255.0, (0, 0), sigmaX=w * 0.014)
    core_norm    = core_buf / 255.0

    result = canvas + glow_blurred * 0.85 + core_norm * 0.90
    result = np.clip(result, 0, 1)

    # ── draw joints — bright dots with glow circles ─────────────────────────
    for j in range(21):
        col = int(np.clip(uvd[j, 0], 4, w-5))
        row = int(np.clip(uvd[j, 1], 4, h-5))
        vel = float(np.clip(joint_vel[j] / 10.0, 0, 1))
        radius = max(4, int(5 + 4 * uvd[j, 2] + 4 * vel))

        # Outer ring
        cv2.circle(result, (col, row), radius + 4, (0.12, 0.12, 0.12), -1)
        # Bright core
        brightness = float(np.clip(0.7 + 0.3 * uvd[j, 2] + 0.5 * vel, 0, 1))
        cv2.circle(result, (col, row), radius, (brightness, brightness, brightness), -1)

    # ── floor shadow below the figure ────────────────────────────────────────
    foot_joints = [13, 14, 15, 18, 19, 20]
    shadow_buf = np.zeros((h, w), np.float32)
    for j in foot_joints:
        col = int(np.clip(uvd[j, 0], 0, w-1))
        row = int(np.clip(uvd[j, 1], 0, h-1))
        cv2.ellipse(shadow_buf, (col, row + 8), (22, 8), 0, 0, 360, (0.35,), -1)
    shadow_blurred = cv2.GaussianBlur(shadow_buf, (0, 0), sigmaX=12)[:, :, np.newaxis]
    result = result * (1 - shadow_blurred * 0.6)

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Velocity computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_joint_velocities(joints_seq: np.ndarray) -> np.ndarray:
    """
    Compute per-frame per-joint speed in mm/frame.
    Returns (T, 21) float32.
    """
    vel = np.zeros((len(joints_seq), 21), np.float32)
    if len(joints_seq) > 1:
        diff = np.diff(joints_seq, axis=0)         # (T-1, 21, 3)
        speed = np.linalg.norm(diff, axis=-1)      # (T-1, 21)
        vel[:-1] = speed
        vel[-1]  = speed[-1]
    return vel


# -- Main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cinematic skeleton visualization (fast, no downloads)")
    parser.add_argument("--query", default="walk forward")
    parser.add_argument("--clips", type=int, default=1, help="How many different clips to render")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width",  type=int, default=720)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--yaw", type=float, default=15.0,
                        help="Camera yaw in degrees (0=front, 30=slight angle)")
    parser.add_argument("--kit-dir", default="data/KIT-ML")
    args = parser.parse_args()

    out_dir = "outputs/videos"
    os.makedirs(out_dir, exist_ok=True)

    clips = find_clips(args.kit_dir, args.query, n=args.clips)
    if not clips:
        # Fall back to a known good clip
        sid = "00964"
        joints, text = load_clip(args.kit_dir, sid)
        clips = [(sid, text, joints)]

    from src.modules.render_engine import RenderEngine, RenderSettings

    for sample_id, text, joints_all in clips:
        log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        log.info("Clip %s — '%s'  (%d raw frames)", sample_id, text, len(joints_all))

        # Subsample
        total = len(joints_all)
        idx = np.linspace(0, total - 1, min(args.frames, total), dtype=int)
        joints_seq = joints_all[idx]   # (F, 21, 3)
        vel_seq    = compute_joint_velocities(joints_seq)   # (F, 21)

        # Pre-build background (same for all frames)
        bg = make_background(args.width, args.height)

        log.info("Rendering %d cinematic frames at %dx%d …",
                 len(joints_seq), args.width, args.height)

        raw_frames = []
        for t, (frame_joints, frame_vel) in enumerate(zip(joints_seq, vel_seq)):
            uvd = project(frame_joints, args.width, args.height, args.yaw)
            frame = render_cinematic_frame(
                uvd, frame_vel, bg, args.width, args.height
            )
            raw_frames.append(frame)

        # M7 post-processing
        log.info("Applying M7 post-processing …")

        class _Frm:
            def __init__(self, rgb): self.rgb = rgb

        settings = RenderSettings(
            motion_blur=True, motion_blur_alpha=0.55,
            dof=False,
            color_grade=True, saturation=1.4, contrast=1.15,
            gamma=0.88, tint=(1.02, 1.0, 0.97),
            vignette=True, vignette_strength=0.55,
            film_grain=False,
        )
        engine = RenderEngine(settings)
        processed = engine._process_frames([_Frm(f) for f in raw_frames])  # noqa: SLF001

        # Write video
        import imageio
        out_path = os.path.join(out_dir, f"cinematic_walk_{sample_id}.mp4")
        writer = imageio.get_writer(out_path, fps=args.fps, codec="libx264",
                                    quality=9, macro_block_size=1)
        for f in processed:
            writer.append_data(f)
        writer.close()
        log.info("Saved → %s", out_path)

    # Open the last video
    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(out_path)])

    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  VISUALISATION COMPLETE")
    print(f"  {len(clips)} clip(s) rendered \u2014 video(s) in {out_dir}/")
    print()
    print("  For photorealistic output (requires ~5.5 GB model download):")
    print("  python examples/demo_human_walk.py --id 00964 --frames 16 --steps 20")
    print("  (let it run overnight alongside M1 training)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

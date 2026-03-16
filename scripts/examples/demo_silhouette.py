#!/usr/bin/env python3
"""Demo: Parametric Human Body Silhouettes — Cinematic Rendering

Generates a walking-cycle skeleton sequence and renders it with
four different body types side-by-side, plus one full cinematic orbit
video to demonstrate the silhouette renderer in action.

Outputs (saved to ``outputs/silhouette_demo/``):
  - ``body_comparison.png``       — 4 body types in T-pose, single frame
  - ``walk_orbit.mp4``            — orbiting camera around walking silhouette
  - ``controlnet_conditioning.png`` — ControlNet-ready conditioning image
  - ``frames/frame_XXXX.png``     — individual orbit frames
"""

from __future__ import annotations

import os
import sys
import math
import logging

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.physics.body_model import BodyParams, BodyType, body_params_from_prompt
from src.modules.physics.silhouette_renderer import (
    SilhouetteSkeletonRenderer,
    SilhouetteProjector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = "outputs/silhouette_demo"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")


# ── Synthetic skeleton generation ────────────────────────────────────────────


def make_t_pose() -> np.ndarray:
    """Return a (21, 3) T-pose skeleton in Y-up mm."""
    j = np.zeros((21, 3), dtype=np.float64)
    j[0]  = [0,    0, 0]        # root / pelvis
    j[1]  = [0,  200, 0]        # spine1
    j[2]  = [0,  500, 0]        # spine2 / chest
    j[3]  = [0,  600, 0]        # neck
    j[4]  = [0,  720, 0]        # head top
    j[5]  = [-200, 550, 0]      # L shoulder
    j[6]  = [-450, 550, 0]      # L elbow
    j[7]  = [-650, 550, 0]      # L wrist
    j[8]  = [200, 550, 0]       # R shoulder
    j[9]  = [450, 550, 0]       # R elbow
    j[10] = [650, 550, 0]       # R wrist
    j[11] = [-100,  -10, 0]     # L hip
    j[12] = [-100, -400, 0]     # L knee
    j[13] = [-100, -800, 0]     # L ankle
    j[14] = [-100, -820,  50]   # L heel
    j[15] = [-100, -820, 130]   # L toe
    j[16] = [100,  -10, 0]      # R hip
    j[17] = [100, -400, 0]      # R knee
    j[18] = [100, -800, 0]      # R ankle
    j[19] = [100, -820,  50]    # R heel
    j[20] = [100, -820, 130]    # R toe
    return j


def generate_walk_cycle(n_frames: int = 72, stride_mm: float = 300.0) -> list[np.ndarray]:
    """Synthesise a looping walk cycle.

    Each frame is a (21, 3) skeleton with sinusoidal leg/arm swing.
    Forward motion is along +Z.
    """
    base = make_t_pose()
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        t = i / n_frames  # 0..1 one full cycle
        phase = 2 * math.pi * t
        pose = base.copy()

        # Forward progression
        z_offset = stride_mm * t * 3
        pose[:, 2] += z_offset

        # Pelvis vertical bob
        bob = 15 * math.sin(2 * phase)
        pose[:, 1] += bob

        # Lateral pelvis sway
        sway = 12 * math.sin(phase)
        pose[:, 0] += sway

        # -- Legs: alternate swing --
        swing_angle = math.sin(phase) * 0.45  # radians
        _swing_leg(pose, "left",  swing_angle)
        _swing_leg(pose, "right", -swing_angle)

        # -- Arms: counter-swing --
        arm_swing = math.sin(phase) * 0.35
        _swing_arm(pose, "left",  -arm_swing)
        _swing_arm(pose, "right",  arm_swing)

        # Slight torso twist
        twist = math.sin(phase) * 0.06
        cos_t, sin_t = math.cos(twist), math.sin(twist)
        for idx in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            x, z = pose[idx, 0], pose[idx, 2] - z_offset
            pose[idx, 0] = cos_t * x - sin_t * z
            pose[idx, 2] = sin_t * x + cos_t * z + z_offset

        frames.append(pose)

    return frames


def _swing_leg(pose: np.ndarray, side: str, angle: float) -> None:
    """Rotate a leg chain about the hip joint in the YZ plane."""
    if side == "left":
        hip, knee, ankle, heel, toe = 11, 12, 13, 14, 15
    else:
        hip, knee, ankle, heel, toe = 16, 17, 18, 19, 20

    origin = pose[hip].copy()
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for j in [knee, ankle, heel, toe]:
        dy = pose[j, 1] - origin[1]
        dz = pose[j, 2] - origin[2]
        pose[j, 1] = origin[1] + cos_a * dy - sin_a * dz
        pose[j, 2] = origin[2] + sin_a * dy + cos_a * dz

    # Bend knee in stance phase
    if angle < 0:
        bend = abs(angle) * 0.4
        k_origin = pose[knee].copy()
        cos_b, sin_b = math.cos(bend), math.sin(bend)
        for j in [ankle, heel, toe]:
            dy = pose[j, 1] - k_origin[1]
            dz = pose[j, 2] - k_origin[2]
            pose[j, 1] = k_origin[1] + cos_b * dy - sin_b * dz
            pose[j, 2] = k_origin[2] + sin_b * dy + cos_b * dz


def _swing_arm(pose: np.ndarray, side: str, angle: float) -> None:
    """Swing arm pendulum-like about the shoulder."""
    if side == "left":
        sho, elb, wri = 5, 6, 7
    else:
        sho, elb, wri = 8, 9, 10

    origin = pose[sho].copy()
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for j in [elb, wri]:
        dy = pose[j, 1] - origin[1]
        dz = pose[j, 2] - origin[2]
        pose[j, 1] = origin[1] + cos_a * dy - sin_a * dz
        pose[j, 2] = origin[2] + sin_a * dy + cos_a * dz


# ── Demo 1: Body type comparison ────────────────────────────────────────────


def demo_body_comparison() -> str:
    """Render 4 different body types side-by-side in T-pose."""
    bodies = [
        ("Slim Woman",       "a slim petite woman"),
        ("Average Person",   "a person walks"),
        ("Athletic Man",     "a tall athletic man"),
        ("Heavy-set Person", "a stocky heavy man"),
    ]

    panels: list[np.ndarray] = []
    for label, prompt in bodies:
        bp = body_params_from_prompt(prompt)
        log.info("  %-20s → height=%.2fm, muscle=%.1f, fat=%.1f, gender=%s",
                 label, bp.height_m, bp.muscle_mass, bp.body_fat, bp.gender)

        renderer = SilhouetteSkeletonRenderer(
            img_w=400, img_h=600,
            yaw_deg=20.0, pitch_deg=-15.0, distance=3.0,
            target=[0.0, 0.0, 0.0],
            body_params=bp,
        )
        frame = renderer.render_frame(make_t_pose())

        # Add label
        cv2.putText(frame, label, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(frame, f"h={bp.height_m:.2f}m  m={bp.muscle_mass:.1f}  f={bp.body_fat:.1f}",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 170, 160), 1, cv2.LINE_AA)
        panels.append(frame)

    composite = np.concatenate(panels, axis=1)
    path = os.path.join(OUT_DIR, "body_comparison.png")
    cv2.imwrite(path, composite)
    log.info("Saved body comparison → %s  (%dx%d)", path, composite.shape[1], composite.shape[0])
    return path


# ── Demo 2: Cinematic orbit video ───────────────────────────────────────────


def demo_walk_orbit() -> str:
    """Render a walking cycle with an orbiting camera → MP4."""
    bp = body_params_from_prompt("a tall athletic man walks forward")
    log.info("Body params: height=%.2fm, muscle=%.1f, gender=%s, type=%s",
             bp.height_m, bp.muscle_mass, bp.gender, bp.body_type.name)

    walk = generate_walk_cycle(n_frames=72)
    n = len(walk)

    renderer = SilhouetteSkeletonRenderer(
        img_w=1280, img_h=720,
        yaw_deg=90.0, pitch_deg=-20.0, distance=4.5,
        target=[0.0, 0.0, 0.0],
        action_label="walk forward",
        body_params=bp,
    )

    frames_rgb: list[np.ndarray] = []
    os.makedirs(FRAMES_DIR, exist_ok=True)

    for i, skeleton in enumerate(walk):
        # Orbit camera 180° over the sequence
        yaw = 90.0 + (i / (n - 1)) * 180.0
        pitch = -20.0 + 5.0 * math.sin(2 * math.pi * i / n)
        dist = 4.5 - 0.5 * math.sin(math.pi * i / n)

        # Track the walking figure (centred on pelvis Z)
        z_centre = float(skeleton[0, 2]) / 1000.0
        target = [0.0, 0.0, z_centre]

        renderer.update_camera(yaw, pitch, dist, target)

        prev = walk[i - 1] if i > 0 else None
        frame = renderer.render_frame(skeleton, prev_xyz=prev)

        # Save individual frame
        frame_path = os.path.join(FRAMES_DIR, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, frame)

        # Convert BGR→RGB for video writer
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Write MP4
    video_path = os.path.join(OUT_DIR, "walk_orbit.mp4")
    writer = imageio.get_writer(video_path, fps=24, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for f in frames_rgb:
        writer.append_data(f)
    writer.close()

    log.info("Saved orbit video → %s  (%d frames, 1280x720)", video_path, n)
    return video_path


# ── Demo 3: ControlNet conditioning image ────────────────────────────────────


def demo_controlnet_conditioning() -> str:
    """Render a silhouette conditioning image (ControlNet-ready, on black bg)."""
    bp = body_params_from_prompt("a muscular man poses")
    projector = SilhouetteProjector(
        img_w=512, img_h=512, cam_yaw_deg=15.0, body_params=bp,
    )

    # Take a mid-walk pose for something interesting
    walk = generate_walk_cycle(n_frames=8)
    pose = walk[2]  # mid-stride

    img = projector.render(pose, body_params=bp)

    path = os.path.join(OUT_DIR, "controlnet_conditioning.png")
    cv2.imwrite(path, img)
    log.info("Saved conditioning image → %s  (512x512)", path)
    return path


# ── Demo 4: Multi-angle comparison ──────────────────────────────────────────


def demo_multi_angle() -> str:
    """Render the same pose from 6 different camera angles."""
    bp = body_params_from_prompt("a tall woman stands")
    skeleton = make_t_pose()

    angles = [
        ("Front",       0.0, -15.0),
        ("3/4 Left",   45.0, -15.0),
        ("Side",       90.0, -15.0),
        ("3/4 Back",  135.0, -15.0),
        ("Back",      180.0, -15.0),
        ("High 3/4",   45.0, -40.0),
    ]

    panels: list[np.ndarray] = []
    for label, yaw, pitch in angles:
        renderer = SilhouetteSkeletonRenderer(
            img_w=320, img_h=480,
            yaw_deg=yaw, pitch_deg=pitch, distance=3.5,
            target=[0.0, 0.0, 0.0],
            body_params=bp,
        )
        frame = renderer.render_frame(skeleton)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2, cv2.LINE_AA)
        panels.append(frame)

    # 2 rows × 3 columns
    row1 = np.concatenate(panels[:3], axis=1)
    row2 = np.concatenate(panels[3:], axis=1)
    composite = np.concatenate([row1, row2], axis=0)

    path = os.path.join(OUT_DIR, "multi_angle.png")
    cv2.imwrite(path, composite)
    log.info("Saved multi-angle grid → %s  (%dx%d)", path, composite.shape[1], composite.shape[0])
    return path


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    log.info("=" * 70)
    log.info("  Parametric Human Body Silhouette — Visual Demo")
    log.info("=" * 70)

    log.info("\n[1/4] Body type comparison (4 body types side-by-side)…")
    p1 = demo_body_comparison()

    log.info("\n[2/4] Cinematic orbit video (walking cycle, 180° orbit)…")
    p2 = demo_walk_orbit()

    log.info("\n[3/4] ControlNet conditioning image (silhouette on black)…")
    p3 = demo_controlnet_conditioning()

    log.info("\n[4/4] Multi-angle comparison (6 camera angles)…")
    p4 = demo_multi_angle()

    log.info("\n" + "=" * 70)
    log.info("  All outputs saved to: %s", os.path.abspath(OUT_DIR))
    log.info("  ─ %s", p1)
    log.info("  ─ %s", p2)
    log.info("  ─ %s", p3)
    log.info("  ─ %s", p4)
    log.info("=" * 70)


if __name__ == "__main__":
    main()

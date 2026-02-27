"""
KIT-ML Walking Demo
====================
Loads a real walking capture from the KIT-ML validation set and renders
it as a 3D stick-figure skeleton inside a natural scene.

No humanoid.urdf joint-driving -- we place each joint sphere directly
from the ground-truth capture positions, so the figure is always upright.

Clip: validation/00964  "a person walks forward."  (3.2 s, looped)

Usage
-----
    python examples/demo_walking.py
    python examples/demo_walking.py --clip 01177 --output outputs/videos/walk_kit.mp4
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import pybullet as p
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KIT-ML joint conventions  (new_joints: T x 21 x 3, mm, Y-up, Z-forward)
# ---------------------------------------------------------------------------
#  0  pelvis (root)        5  l_shoulder     11  l_hip
#  1  spine1               6  l_elbow        12  l_knee
#  2  spine2               7  l_wrist        13  l_ankle
#  3  neck                 8  r_shoulder     14  l_toe
#  4  head                 9  r_elbow        16  r_hip
#                         10  r_wrist        17  r_knee
#                                            18  r_ankle
#                                            19  r_toe
#                                            15  unused (index gap)
#                                            20  extra
BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # spine to head
    (3, 5), (5, 6), (6, 7),                 # left arm
    (3, 8), (8, 9), (9, 10),                # right arm
    (0, 11), (11, 12), (12, 13), (13, 14),  # left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # right leg
]

R_JOINT = 0.025   # metre  joint sphere radius
R_BONE  = 0.015   # metre  bone capsule radius


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def kit_to_pb(pos_mm: np.ndarray) -> np.ndarray:
    """KIT (mm, Y-up, Z-forward)  ->  PyBullet (m, Z-up, Y-forward)."""
    if pos_mm.ndim == 1:
        return np.array([pos_mm[0] * 1e-3, pos_mm[2] * 1e-3, pos_mm[1] * 1e-3])
    out = np.zeros_like(pos_mm)
    out[:, 0] = pos_mm[:, 0] * 1e-3   # X  lateral
    out[:, 1] = pos_mm[:, 2] * 1e-3   # Z  -> Y (forward)
    out[:, 2] = pos_mm[:, 1] * 1e-3   # Y  -> Z (up)
    return out


def load_clip(kit_dir: str, clip_id: str) -> np.ndarray:
    """Load KIT-ML clip -> (T, 21, 3) array in PyBullet metres, Z-up."""
    path = pathlib.Path(kit_dir) / "new_joints" / f"{clip_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"KIT-ML clip not found: {path}")
    raw = np.load(path)           # (T, 21, 3) mm
    frames = np.zeros_like(raw)
    frames[:, :, 0] = raw[:, :, 0] * 1e-3
    frames[:, :, 1] = raw[:, :, 2] * 1e-3   # Z -> Y (forward)
    frames[:, :, 2] = raw[:, :, 1] * 1e-3   # Y -> Z (up)
    # centre XY so the clip starts at origin
    frames[:, :, 0] -= frames[0, 0, 0]
    frames[:, :, 1] -= frames[0, 0, 1]
    return frames


def loop_clip(frames: np.ndarray, n_loops: int) -> np.ndarray:
    """Tile clip n_loops times, shifting root position continuosly."""
    delta = frames[-1, 0, :] - frames[0, 0, :]
    parts: list[np.ndarray] = []
    offset = np.zeros(3)
    for _ in range(n_loops):
        part = frames.copy()
        part[:, :, 0] += offset[0]
        part[:, :, 1] += offset[1]
        parts.append(part)
        offset += delta
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------

def quat_align_z_to(direction: np.ndarray) -> Tuple[float, float, float, float]:
    """Return (x,y,z,w) quaternion rotating Z-axis onto `direction`."""
    d = direction / (np.linalg.norm(direction) + 1e-8)
    z = np.array([0.0, 0.0, 1.0])
    dot = float(np.dot(z, d))
    if dot > 0.9999:
        return (0.0, 0.0, 0.0, 1.0)
    if dot < -0.9999:
        return (1.0, 0.0, 0.0, 0.0)
    axis  = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(angle * 0.5)
    c = math.cos(angle * 0.5)
    return (float(axis[0]*s), float(axis[1]*s), float(axis[2]*s), float(c))


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(client: int) -> None:
    """Ground plane + path tiles + trees + park bench."""
    p.loadURDF("plane.urdf", physicsClientId=client)

    # path tiles (alternating brightness every 1 m)
    for i in range(22):
        y    = i * 1.0
        tone = 0.52 + 0.08 * (i % 2)
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.38, 0.48, 0.003],
                                    physicsClientId=client)
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.38, 0.48, 0.003],
                                 rgbaColor=[tone, tone + 0.12, tone, 1.0],
                                 physicsClientId=client)
        p.createMultiBody(0, cs, vs, basePosition=[0.0, y, 0.003],
                          physicsClientId=client)

    # trees (trunk cylinder + canopy sphere)
    trees = [
        (-2.2, 2.0), ( 2.6, 3.5), (-2.6, 5.5), ( 2.0, 7.0),
        (-2.2, 9.5), ( 2.6,11.0), (-2.6,13.5), ( 2.0,16.0),
    ]
    for tx, ty in trees:
        # trunk
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=2.2,
                                    physicsClientId=client)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.12, length=2.2,
                                 rgbaColor=[0.42, 0.28, 0.14, 1.0],
                                 physicsClientId=client)
        p.createMultiBody(0, tc, tv, basePosition=[tx, ty, 1.1],
                          physicsClientId=client)
        # canopy
        cc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.80,
                                    physicsClientId=client)
        cv = p.createVisualShape(p.GEOM_SPHERE, radius=0.80,
                                 rgbaColor=[0.18, 0.52, 0.16, 0.92],
                                 physicsClientId=client)
        p.createMultiBody(0, cc, cv, basePosition=[tx, ty, 3.0],
                          physicsClientId=client)

    # park bench
    bc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.22],
                                physicsClientId=client)
    bv = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.2, 0.22],
                             rgbaColor=[0.50, 0.35, 0.15, 1.0],
                             physicsClientId=client)
    p.createMultiBody(0, bc, bv, basePosition=[-1.9, 4.5, 0.22],
                      physicsClientId=client)


# ---------------------------------------------------------------------------
# Skeleton creation / update
# ---------------------------------------------------------------------------

def create_skeleton(client: int) -> Tuple[List[int], List[int]]:
    """Create 21 joint spheres + 18 bone capsules. Returns (joint_ids, bone_ids)."""
    joint_ids: List[int] = []
    bone_ids:  List[int] = []

    for _ in range(21):
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=R_JOINT,
                                    physicsClientId=client)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=R_JOINT,
                                 rgbaColor=[0.95, 0.72, 0.60, 1.0],
                                 physicsClientId=client)
        jid = p.createMultiBody(0, cs, vs, basePosition=[0, 0, -10],
                                physicsClientId=client)
        joint_ids.append(jid)

    for _ in BONES:
        rc = p.createCollisionShape(p.GEOM_CAPSULE, radius=R_BONE, height=0.25,
                                    physicsClientId=client)
        rv = p.createVisualShape(p.GEOM_CAPSULE, radius=R_BONE, length=0.25,
                                 rgbaColor=[0.55, 0.65, 0.88, 1.0],
                                 physicsClientId=client)
        bid = p.createMultiBody(0, rc, rv, basePosition=[0, 0, -10],
                                physicsClientId=client)
        bone_ids.append(bid)

    return joint_ids, bone_ids


def update_skeleton(joints: np.ndarray,
                    joint_ids: List[int],
                    bone_ids: List[int],
                    client: int) -> None:
    """Teleport all skeleton elements to current-frame positions."""
    for i, jid in enumerate(joint_ids):
        p.resetBasePositionAndOrientation(
            jid, joints[i].tolist(), [0, 0, 0, 1], physicsClientId=client
        )
    for k, (a_idx, b_idx) in enumerate(BONES):
        a = joints[a_idx]
        b = joints[b_idx]
        mid = ((a + b) * 0.5).tolist()
        direction = b - a
        if np.linalg.norm(direction) < 1e-4:
            continue
        orn = quat_align_z_to(direction)
        p.resetBasePositionAndOrientation(
            bone_ids[k], mid, list(orn), physicsClientId=client
        )


# ---------------------------------------------------------------------------
# Per-joint colour overrides
# ---------------------------------------------------------------------------

JOINT_COLOURS = {
    4:  [1.00, 0.85, 0.70, 1.0],   # head
    3:  [0.95, 0.80, 0.65, 1.0],   # neck
    13: [0.25, 0.18, 0.10, 1.0],   # l_ankle
    14: [0.25, 0.18, 0.10, 1.0],   # l_toe
    18: [0.25, 0.18, 0.10, 1.0],   # r_ankle
    19: [0.25, 0.18, 0.10, 1.0],   # r_toe
}


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(
    kit_dir:     str = "data/KIT-ML",
    clip_id:     str = "00964",
    n_loops:     int = 3,
    fps:         int = 30,
    output_path: str = "outputs/videos/walking_kit.mp4",
) -> str:
    import imageio

    # -- load & prepare motion --
    frames_raw = load_clip(kit_dir, clip_id)
    frames     = loop_clip(frames_raw, n_loops)
    n_frames   = len(frames)
    duration   = n_frames / 20.0
    log.info("Clip %s | %d motion frames @ 20 fps = %.1f s  (%d loops)",
             clip_id, n_frames, duration, n_loops)

    # -- pybullet --
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    p.setTimeStep(1.0 / 240.0, physicsClientId=client)

    build_scene(client)
    joint_ids, bone_ids = create_skeleton(client)

    for jidx, rgba in JOINT_COLOURS.items():
        p.changeVisualShape(joint_ids[jidx], -1, rgbaColor=rgba,
                            physicsClientId=client)

    # -- render --
    W, H    = 960, 540
    n_video = int(duration * fps)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                macro_block_size=None)

    log.info("Rendering %d frames ...", n_video)
    for vi in range(n_video):
        t       = vi / fps
        kit_idx = int(t * 20) % n_frames
        joints  = frames[kit_idx]          # (21, 3) metres, Z-up

        update_skeleton(joints, joint_ids, bone_ids, client)
        p.stepSimulation(physicsClientId=client)

        root = joints[0]
        cam_target = [float(root[0]),
                      float(root[1]) + 0.2,
                      float(root[2]) + 0.05]
        cam_yaw   = 55 - 50 * math.sin(2 * math.pi * t / duration)
        cam_pitch = -18 - 4 * math.sin(math.pi * t / duration)
        cam_dist  = 4.0 - 0.7 * math.sin(math.pi * t / duration)

        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=cam_dist, yaw=cam_yaw, pitch=cam_pitch,
            roll=0, upAxisIndex=2,
        )
        proj = p.computeProjectionMatrixFOV(
            fov=52, aspect=W / H, nearVal=0.1, farVal=100.0
        )
        _, _, rgb, _, _ = p.getCameraImage(
            width=W, height=H,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=client,
        )
        img = np.array(rgb, dtype=np.uint8).reshape(H, W, 4)[:, :, :3]
        writer.append_data(img)

        if vi % (fps * 2) == 0:
            log.info("  frame %d / %d  (t=%.1f s)", vi, n_video, t)

    writer.close()
    p.disconnect(client)
    log.info("Done: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="KIT-ML walking demo (stick figure)")
    ap.add_argument("--kit-dir", default="data/KIT-ML", help="KIT-ML dataset root")
    ap.add_argument("--clip",    default="00964",
                    help="Clip ID from val/test set (default: 00964 'walks forward')")
    ap.add_argument("--loops",   type=int, default=3,
                    help="Times to loop the clip (default: 3)")
    ap.add_argument("--fps",     type=int, default=30)
    ap.add_argument("--output",  default="outputs/videos/walking_kit.mp4")
    return ap.parse_args()


if __name__ == "__main__":
    args = _args()
    path = run_demo(
        kit_dir=args.kit_dir,
        clip_id=args.clip,
        n_loops=args.loops,
        fps=args.fps,
        output_path=args.output,
    )
    print(f"Video: {path}")

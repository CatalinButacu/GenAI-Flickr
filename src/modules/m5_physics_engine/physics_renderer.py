"""
#WHERE
    Used by pipeline.py (M5+M7 integration) for cinematic skeleton rendering.

#WHAT
    Physics-verified skeleton renderer — reads 3D joint positions from PyBullet
    after physics simulation and renders them as a cinematic skeleton overlay
    (anti-aliased bones, joint circles, bone coloring).

#INPUT
    HumanoidBody with link positions, VirtualCamera config, frame size.

#OUTPUT
    RGBA numpy arrays with rendered skeleton frames.

Rendering pipeline per frame
----------------------------
1. ``link_positions = humanoid.get_link_world_positions()``    (PyBullet)
2. ``xyz_21 = physics_links_to_skeleton(link_positions)``     (remapping)
3. ``uvd    = project_3d_to_screen(xyz_21, camera, w, h)``    (virtual cam)
4. ``frame  = render_skeleton_glow(uvd, velocity, bg, w, h)`` (cinematic)
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# ── Skeleton bone connectivity (21 KIT-ML joints) ────────────────────────────
# Each tuple: (parent_idx, child_idx)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # spine + head
    (2, 5), (5, 6), (6, 7),                   # left arm
    (2, 8), (8, 9), (9, 10),                  # right arm
    (0, 11), (0, 16), (11, 16),               # pelvis → hips
    (11, 12), (12, 13), (13, 14), (14, 15),   # left leg
    (16, 17), (17, 18), (18, 19), (19, 20),   # right leg
    (5, 8),                                   # shoulder girdle
]

# HSV colour per bone (OpenCV: H 0-179, S 0-255, value multiplier)
_BONE_HSV: List[Tuple[int, int, float]] = [
    (30,  180, 1.0),  # spine
    (30,  180, 1.0),
    (30,  180, 1.0),
    (30,  180, 1.0),
    (100, 220, 1.0),  # L arm
    (100, 220, 1.0),
    (100, 220, 1.0),
    (10,  240, 1.0),  # R arm
    (10,  240, 1.0),
    (10,  240, 1.0),
    (60,  160, 0.8),  # pelvis
    (60,  160, 0.8),
    (60,  160, 0.8),
    (140, 200, 1.0),  # L leg
    (140, 200, 1.0),
    (140, 200, 1.0),
    (140, 200, 1.0),
    (130, 200, 1.0),  # R leg
    (130, 200, 1.0),
    (130, 200, 1.0),
    (130, 200, 1.0),
    (0,   200, 0.9),  # shoulder girdle
]


def _hsv_bgr(h: int, s: int, v_frac: float) -> Tuple[float, float, float]:
    """OpenCV HSV → BGR float 0-255."""
    px = np.array([[[h, s, int(255 * v_frac)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return float(bgr[0]), float(bgr[1]), float(bgr[2])


# ── PyBullet link → KIT-ML joint mapping ─────────────────────────────────────
# humanoid.urdf joint names (from HumanoidBody._joint_info) mapped to
# their best KIT-ML counterpart index.
# Joints present:  abdomen_x/y/z  r/l_hip_x/y/z  r/l_knee  r/l_ankle
#                  r/l_shoulder_x/y  r/l_elbow
_LINK_TO_KIT: Dict[str, int] = {
    "base":           0,   # pelvis / root
    "abdomen_x":      1,   # lower spine
    "abdomen_y":      1,
    "abdomen_z":      2,   # chest
    "right_hip_x":   16,
    "right_hip_y":   16,
    "right_hip_z":   16,
    "right_knee":    17,
    "right_ankle":   18,
    "left_hip_x":    11,
    "left_hip_y":    11,
    "left_hip_z":    11,
    "left_knee":     12,
    "left_ankle":    13,
    "right_shoulder_x":  8,
    "right_shoulder_y":  8,
    "right_elbow":       9,
    "left_shoulder_x":   5,
    "left_shoulder_y":   5,
    "left_elbow":        6,
}


def _extrapolate_missing_joints(
    xyz: np.ndarray, counts: np.ndarray
) -> None:
    """Fill in joints not directly mapped from PyBullet links."""
    # Neck and head: extend spine
    if counts[3] == 0 and counts[2] > 0 and counts[1] > 0:
        spine_dir = xyz[2] - xyz[1]
        xyz[3] = xyz[2] + spine_dir * 0.6
        xyz[4] = xyz[2] + spine_dir * 1.1
    # Wrists: extend from elbows
    if counts[7] == 0 and counts[6] > 0 and counts[5] > 0:
        xyz[7] = xyz[6] + (xyz[6] - xyz[5]) * 0.7
    if counts[10] == 0 and counts[9] > 0 and counts[8] > 0:
        xyz[10] = xyz[9] + (xyz[9] - xyz[8]) * 0.7
    # Left toes
    if counts[14] == 0 and counts[13] > 0 and counts[12] > 0:
        fwd = xyz[13] - xyz[12]
        fwd[1] = 0.0
        xyz[14] = xyz[13] + fwd * 0.3 + np.array([0, -0.05, 0])
        xyz[15] = xyz[13] + fwd * 0.5 + np.array([0, -0.05, 0])
    # Right toes
    if counts[19] == 0 and counts[18] > 0 and counts[17] > 0:
        fwd = xyz[18] - xyz[17]
        fwd[1] = 0.0
        xyz[19] = xyz[18] + fwd * 0.3 + np.array([0, -0.05, 0])
        xyz[20] = xyz[18] + fwd * 0.5 + np.array([0, -0.05, 0])


def physics_links_to_skeleton(
    link_positions: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Convert PyBullet link world positions to a (21, 3) skeleton array.

    Coordinates are converted from PyBullet Z-up metres to Y-up metres
    so the cinematic projector (which was written for KIT-ML Y-up) works.

    Missing joints (e.g. wrists, toes, head) are extrapolated from their
    parent links.  This is sufficient for a convincing skeleton render.
    """
    xyz = np.zeros((21, 3), dtype=np.float64)
    counts = np.zeros(21, dtype=np.int32)

    for link_name, pos in link_positions.items():
        if link_name not in _LINK_TO_KIT:
            continue
        idx = _LINK_TO_KIT[link_name]
        # Z-up (PyBullet) → Y-up: swap y and z
        xyz[idx] += np.array([pos[0], pos[2], pos[1]], dtype=np.float64)
        counts[idx] += 1

    # Average multiple links mapped to the same KIT index
    for i in range(21):
        if counts[i] > 1:
            xyz[i] /= counts[i]

    _extrapolate_missing_joints(xyz, counts)
    return xyz * 1000.0


# ── 3-D → 2-D projection ─────────────────────────────────────────────────────

class _VirtualCamera:
    """
    Perspective camera matching PyBullet's view matrix for layout consistency.
    Independent of PyBullet so we can render without ER_TINY_RENDERER.
    """

    def __init__(
        self,
        target: Optional[np.ndarray] = None,
        distance: float = 3.5,
        yaw_deg: float = 30.0,
        pitch_deg: float = -20.0,
        fov_deg: float = 55.0,
        img_w: int = 720,
        img_h: int = 1080,
    ):
        self.target    = np.array(target if target is not None else [0, 0, 1.0])
        self.distance  = distance
        self.yaw       = np.radians(yaw_deg)
        self.pitch     = np.radians(pitch_deg)
        self.fov       = np.radians(fov_deg)
        self.w         = img_w
        self.h         = img_h
        self._update_extrinsics()

    def update(self, yaw_deg: float, pitch_deg: float,
               distance: float, target: list) -> None:
        self.yaw      = np.radians(yaw_deg)
        self.pitch    = np.radians(pitch_deg)
        self.distance = distance
        self.target   = np.array(target)
        self._update_extrinsics()

    def _update_extrinsics(self) -> None:
        cy, sy = np.cos(self.yaw),   np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        # Camera position (Y-up world — but PyBullet is Z-up — map accordingly)
        # We use a Z-up world camera here because PyBullet skeleton is Z-up
        # and physics_links_to_skeleton already converted to Y-up mm.
        # The input xyz from physics_links_to_skeleton is in Y-up mm,
        # so this camera is Y-up.
        eye_x = self.target[0] + self.distance * cp * sy
        eye_y = self.target[1] + self.distance * sp           # Y-up height
        eye_z = self.target[2] + self.distance * cp * cy
        eye   = np.array([eye_x, eye_y, eye_z])

        forward = self.target - eye
        forward /= np.linalg.norm(forward) + 1e-9
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0.0, 0.0, 1.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        cam_up = np.cross(right, forward)

        self._eye     = eye
        self._right   = right
        self._cam_up  = cam_up
        self._forward = forward

    def project(self, xyz_mm: np.ndarray) -> np.ndarray:
        """
        Project (N, 3) Y-up mm world coordinates to (N, 3) screen coords.
        Returns (col, row, depth_norm).
        """
        pts = xyz_mm / 1000.0   # mm → m
        # Camera space
        diff  = pts - self._eye
        xc    = np.dot(diff, self._right)
        yc    = np.dot(diff, self._cam_up)
        zc    = np.dot(diff, self._forward)

        # Perspective divide
        zc_safe = np.where(np.abs(zc) > 1e-4, zc, 1e-4)
        f    = 1.0 / np.tan(self.fov / 2.0)
        col  = (xc / zc_safe) * f * (self.w / 2.0) + self.w / 2.0
        row  = -(yc / zc_safe) * f * (self.w / 2.0) + self.h / 2.0

        depth_norm = np.clip((zc - zc.min()) / (zc.max() - zc.min() + 1e-6), 0, 1)
        return np.stack([col, row, depth_norm], axis=-1)


# ── Cinematic skeleton rendering ──────────────────────────────────────────────

def _make_background(w: int, h: int) -> np.ndarray:
    """Dark studio background with a warm ground glow."""
    cx, cy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r  = np.sqrt(cx**2 + cy**2)
    bg = np.clip(1 - 0.75 * r, 0.03, 0.18)

    # Ground glow — subtle warm zone at bottom
    hy = int(h * 0.70)
    glow = np.zeros((h, w), np.float32)
    glow[hy:, :] = np.linspace(0, 1, h - hy)[:, None] ** 1.8 * 0.18
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=w * 0.06)

    rgb = np.stack([
        np.clip(bg + glow * 0.30, 0, 1),
        np.clip(bg + glow * 0.20, 0, 1),
        np.clip(bg + glow * 0.05, 0, 1),
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _draw_bone_glow(
    canvas: np.ndarray,
    glow_buf: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    bgr: Tuple[float, float, float],
    thickness: int,
) -> None:
    cv2.line(canvas,   pt1, pt2, bgr, thickness, cv2.LINE_AA)
    cv2.line(glow_buf, pt1, pt2, bgr, thickness + 10, cv2.LINE_AA)


def render_physics_skeleton_frame(
    uvd: np.ndarray,
    joint_vel: np.ndarray,
    bg: np.ndarray,
    w: int,
    h: int,
    camera: Optional["_VirtualCamera"] = None,
    action_label: str = "",
) -> np.ndarray:
    """
    Render one cinematic skeleton frame.

    uvd       : (21, 3)  — (col, row, depth_norm) from VirtualCamera.project()
    joint_vel : (21,)    — per-joint speed (mm/frame) for heat colouring
    bg        : (H, W, 3) uint8 background
    camera    : _VirtualCamera instance for drawing ground grid
    action_label : text label drawn at top-right corner
    """
    canvas   = bg.astype(np.float32) / 255.0

    # Draw ground grid first (behind skeleton)
    if camera is not None:
        _draw_ground_grid(canvas, camera, w, h)

    glow_buf = np.zeros((h, w, 3), np.float32)
    core_buf = np.zeros((h, w, 3), np.float32)

    for bone_idx, (ja, jb) in enumerate(BONES):
        hue, sat, val_m = _BONE_HSV[bone_idx]

        depth_mid = float((uvd[ja, 2] + uvd[jb, 2]) * 0.5)
        vel_avg   = float(np.clip((joint_vel[ja] + joint_vel[jb]) * 0.5 / 80.0, 0, 1))

        v  = float(np.clip(val_m * (0.55 + 0.45 * depth_mid + 0.30 * vel_avg), 0, 1))
        s  = int(np.clip(sat * (0.7 + 0.3 * vel_avg), 0, 255))

        bgr  = _hsv_bgr(hue, s, v)
        pt_a = (int(np.clip(uvd[ja, 0], 0, w - 1)),
                int(np.clip(uvd[ja, 1], 0, h - 1)))
        pt_b = (int(np.clip(uvd[jb, 0], 0, w - 1)),
                int(np.clip(uvd[jb, 1], 0, h - 1)))

        thick = max(2, int(2 + 5 * depth_mid + 3 * vel_avg))
        _draw_bone_glow(core_buf, glow_buf, pt_a, pt_b, bgr, thick)

    # Bloom: blur glow layer and composite
    sigma = max(1.0, w * 0.012)
    glow_blurred = cv2.GaussianBlur(glow_buf / 255.0, (0, 0), sigmaX=sigma)
    result = canvas + glow_blurred * 0.90 + core_buf / 255.0 * 0.85

    # Joints
    for j in range(21):
        col = int(np.clip(uvd[j, 0], 4, w - 5))
        row = int(np.clip(uvd[j, 1], 4, h - 5))
        vel = float(np.clip(joint_vel[j] / 80.0, 0, 1))
        radius = max(4, int(5 + 5 * float(uvd[j, 2]) + 5 * vel))

        bright = float(np.clip(0.7 + 0.3 * float(uvd[j, 2]) + 0.5 * vel, 0, 1))
        cv2.circle(result, (col, row), radius + 4, (0.10, 0.10, 0.10), -1)
        cv2.circle(result, (col, row), radius,     (bright, bright, bright), -1)

    # Floor shadow under feet
    for j in [13, 14, 15, 18, 19, 20]:
        col = int(np.clip(uvd[j, 0], 0, w - 1))
        row = int(np.clip(uvd[j, 1], 0, h - 1))
        shadow = np.zeros((h, w), np.float32)
        cv2.ellipse(shadow, (col, row + 10), (28, 9), 0, 0, 360, (0.40,), -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=14)
        result = result * (1 - shadow[:, :, np.newaxis] * 0.55)

    # Action label overlay
    if action_label:
        _draw_action_label(result, action_label, w, h)

    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


def _draw_ground_grid(
    canvas: np.ndarray,
    camera: "_VirtualCamera",
    w: int, h: int,
    grid_extent: float = 3000.0,
    grid_step: float = 500.0,
) -> None:
    """
    Draw a perspective ground grid on the canvas for spatial orientation.
    Grid is on the XZ plane at Y=0 (floor level), in Y-up mm coordinates.
    """
    lines = []
    n = int(grid_extent / grid_step)
    for i in range(-n, n + 1):
        coord = i * grid_step
        # Lines parallel to Z axis
        lines.append(np.array([[coord, 0.0, -grid_extent],
                                [coord, 0.0, grid_extent]]))
        # Lines parallel to X axis
        lines.append(np.array([[-grid_extent, 0.0, coord],
                                [grid_extent, 0.0, coord]]))

    for seg in lines:
        uv_a = camera.project(seg[0:1])[0]
        uv_b = camera.project(seg[1:2])[0]
        # Skip lines behind camera or far off-screen
        if uv_a[2] < 0 or uv_b[2] < 0:
            continue
        pa = (int(uv_a[0]), int(uv_a[1]))
        pb = (int(uv_b[0]), int(uv_b[1]))
        # Clip to frame with margin
        margin = w * 2
        if (pa[0] < -margin and pb[0] < -margin) or \
           (pa[0] > w + margin and pb[0] > w + margin):
            continue
        if (pa[1] < -margin and pb[1] < -margin) or \
           (pa[1] > h + margin and pb[1] > h + margin):
            continue
        # Centre axis lines slightly brighter
        is_axis = abs(seg[0][0]) < 1.0 or abs(seg[0][2]) < 1.0
        colour = (0.18, 0.22, 0.25) if is_axis else (0.10, 0.12, 0.14)
        thick = 2 if is_axis else 1
        cv2.line(canvas, pa, pb, colour, thick, cv2.LINE_AA)


def _draw_action_label(
    canvas: np.ndarray,
    action_label: str,
    w: int, h: int,
) -> None:
    """Draw the current action name on the frame for clarity."""
    if not action_label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 1200.0)
    thick = max(1, int(scale * 2))
    text = action_label.upper()
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = w - tw - 20
    y = th + 20
    # Dark background bar
    cv2.rectangle(canvas, (x - 10, 2), (w - 2, y + 10),
                  (0.06, 0.06, 0.06), -1)
    # Label text in warm white
    cv2.putText(canvas, text, (x, y), font, scale,
                (0.85, 0.80, 0.70), thick, cv2.LINE_AA)


# ── Main interface ────────────────────────────────────────────────────────────

class PhysicsSkeletonRenderer:
    """
    Renders physics-verified skeleton frames with perspective ground grid,
    action label, and clear spatial orientation.

    Usage::

        renderer = PhysicsSkeletonRenderer(img_w=1280, img_h=720)
        for step in simulation_loop:
            link_pos = humanoid.get_link_world_positions()
            xyz_21   = physics_links_to_skeleton(link_pos)
            frame_rgb = renderer.render_frame(xyz_21, prev_xyz_21)
    """

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        yaw_deg: float = 45.0,
        pitch_deg: float = -25.0,
        distance: float = 4.0,
        target: Optional[List[float]] = None,
        action_label: str = "",
    ):
        self.w   = img_w
        self.h   = img_h
        self.action_label = action_label
        self.cam = _VirtualCamera(
            target=np.array(target or [0.0, 900.0, 0.0]),
            distance=distance,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            img_w=img_w,
            img_h=img_h,
        )
        self._bg       = _make_background(img_w, img_h)
        self._prev_xyz = None

    def update_camera(self, yaw_deg: float, pitch_deg: float,
                      distance: float, target: list) -> None:
        self.cam.update(yaw_deg, pitch_deg, distance, target)

    def render_frame(
        self,
        xyz_21: np.ndarray,
        prev_xyz: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        xyz_21   : (21, 3) physics-verified joint positions, Y-up mm
        prev_xyz : previous frame's xyz for velocity colouring (optional)
        """
        if prev_xyz is None:
            prev_xyz = self._prev_xyz if self._prev_xyz is not None else xyz_21

        vel = np.linalg.norm(xyz_21 - prev_xyz, axis=-1).astype(np.float32)
        self._prev_xyz = xyz_21.copy()

        uvd = self.cam.project(xyz_21)
        frame = render_physics_skeleton_frame(
            uvd, vel, self._bg, self.w, self.h,
            camera=self.cam, action_label=self.action_label,
        )
        return frame

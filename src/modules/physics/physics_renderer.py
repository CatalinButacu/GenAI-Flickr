"""Skeleton utilities — PyBullet link extraction, auto-orient, virtual camera.

Used by the physics and rendering stages. The old cinematic stick-figure
renderer (PhysicsSkeletonRenderer) has been replaced by aitviewer SMPL
rendering; only the utility functions remain.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from src.shared.constants import HUMANOID_BONES as BONES

log = logging.getLogger(__name__)

# ── PyBullet link → 21-joint humanoid mapping ────────────────────────────────

_LINK_TO_KIT: dict[str, int] = {
    "base": 0, "root": 1, "chest": 2, "neck": 3,
    "left_shoulder": 5, "left_elbow": 6, "left_wrist": 7,
    "right_shoulder": 8, "right_elbow": 9, "right_wrist": 10,
    "left_hip": 11, "left_knee": 12, "left_ankle": 13,
    "right_hip": 16, "right_knee": 17, "right_ankle": 18,
}


def _extrapolate_missing_joints(xyz: np.ndarray, counts: np.ndarray) -> None:
    """Fill in joints not directly mapped from PyBullet links."""
    if counts[4] == 0 and counts[3] > 0 and counts[2] > 0:
        xyz[4] = xyz[3] + (xyz[3] - xyz[2]) * 0.5
    if counts[7] == 0 and counts[6] > 0 and counts[5] > 0:
        xyz[7] = xyz[6] + (xyz[6] - xyz[5]) * 0.7
    if counts[10] == 0 and counts[9] > 0 and counts[8] > 0:
        xyz[10] = xyz[9] + (xyz[9] - xyz[8]) * 0.7
    for ankle, knee, heel, toe in [(13, 12, 14, 15), (18, 17, 19, 20)]:
        if counts[heel] == 0 and counts[ankle] > 0 and counts[knee] > 0:
            fwd = xyz[ankle] - xyz[knee]
            fwd[1] = 0.0
            n = np.linalg.norm(fwd)
            if n > 1e-3:
                fwd /= n
            xyz[heel] = xyz[ankle] + fwd * 0.080
            xyz[toe] = xyz[ankle] + fwd * 0.140


def physics_links_to_skeleton(link_positions: dict[str, np.ndarray]) -> np.ndarray:
    """Convert PyBullet link positions to a (21, 3) skeleton (Y-up, mm)."""
    xyz = np.zeros((21, 3), dtype=np.float64)
    counts = np.zeros(21, dtype=np.int32)
    for name, pos in link_positions.items():
        if name not in _LINK_TO_KIT:
            continue
        idx = _LINK_TO_KIT[name]
        xyz[idx] += np.array([pos[0], pos[2], pos[1]], dtype=np.float64)
        counts[idx] += 1
    for i in range(21):
        if counts[i] > 1:
            xyz[i] /= counts[i]
    _extrapolate_missing_joints(xyz, counts)
    return xyz * 1000.0


def auto_orient_skeleton(skeleton_seq: list[np.ndarray]) -> list[np.ndarray]:
    """Ensure head is above feet on Y axis for each frame."""
    if not skeleton_seq:
        return skeleton_seq
    result = []
    for xyz in skeleton_seq:
        out = xyz.copy()
        if out[4, 1] - (out[13, 1] + out[18, 1]) / 2.0 < 0:
            out[:, 1] *= -1
        result.append(out)
    if result:
        h = result[0][4, 1]
        f = (result[0][13, 1] + result[0][18, 1]) / 2.0
        log.info("[M5] auto-orient: height=%.1fm", (h - f) / 1000.0)
    return result


# ── Virtual camera (used by silhouette renderer) ─────────────────────────────

class _VirtualCamera:
    """Perspective camera for 3D→2D projection (Y-up, mm coordinates)."""

    def __init__(self, target=None, distance=3.5, yaw_deg=30.0,
                 pitch_deg=-20.0, fov_deg=55.0, img_w=720, img_h=1080):
        self.target = np.array(target if target is not None else [0, 0, 1.0])
        self.distance = distance
        self.yaw = np.radians(yaw_deg)
        self.pitch = np.radians(pitch_deg)
        self.fov = np.radians(fov_deg)
        self.w, self.h = img_w, img_h
        self._update()

    def update(self, yaw_deg, pitch_deg, distance, target):
        self.yaw = np.radians(yaw_deg)
        self.pitch = np.radians(pitch_deg)
        self.distance = distance
        self.target = np.array(target)
        self._update()

    def _update(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        eye = self.target + [self.distance * cp * sy,
                             -self.distance * sp,
                             self.distance * cp * cy]
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd) + 1e-9
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        self._eye, self._right = eye, right
        self._cam_up = np.cross(right, fwd)
        self._forward = fwd

    def project(self, xyz_mm: np.ndarray) -> np.ndarray:
        """Project (N, 3) Y-up mm coords → (N, 3) screen [col, row, depth]."""
        pts = xyz_mm / 1000.0
        diff = pts - self._eye
        xc = np.dot(diff, self._right)
        yc = np.dot(diff, self._cam_up)
        zc = np.dot(diff, self._forward)
        zc_safe = np.where(np.abs(zc) > 1e-4, zc, 1e-4)
        f = 1.0 / np.tan(self.fov / 2.0)
        col = (xc / zc_safe) * f * (self.w / 2.0) + self.w / 2.0
        row = -(yc / zc_safe) * f * (self.w / 2.0) + self.h / 2.0
        depth = np.clip((zc - zc.min()) / (zc.max() - zc.min() + 1e-6), 0, 1)
        return np.stack([col, row, depth], axis=-1)


# ── Background and grid helpers (used by silhouette renderer) ─────────────────

def _make_background(w: int, h: int) -> np.ndarray:
    """Dark studio background with warm ground glow."""
    cx, cy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    bg = np.clip(1 - 0.75 * np.sqrt(cx**2 + cy**2), 0.03, 0.18)
    hy = int(h * 0.70)
    glow = np.zeros((h, w), np.float32)
    glow[hy:] = np.linspace(0, 1, h - hy)[:, None] ** 1.8 * 0.18
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=w * 0.06)
    rgb = np.stack([
        np.clip(bg + glow * 0.30, 0, 1),
        np.clip(bg + glow * 0.20, 0, 1),
        np.clip(bg + glow * 0.05, 0, 1),
    ], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _draw_ground_grid(canvas, camera, w, h,
                      extent=3000.0, step=500.0):
    """Draw perspective ground grid on XZ plane."""
    n = int(extent / step)
    margin = w * 2
    for i in range(-n, n + 1):
        coord = i * step
        for seg in [np.array([[coord, 0, -extent], [coord, 0, extent]]),
                    np.array([[-extent, 0, coord], [extent, 0, coord]])]:
            uv_a = camera.project(seg[0:1])[0]
            uv_b = camera.project(seg[1:2])[0]
            if uv_a[2] < 0 or uv_b[2] < 0:
                continue
            pa = (int(uv_a[0]), int(uv_a[1]))
            pb = (int(uv_b[0]), int(uv_b[1]))
            if (pa[0] < -margin and pb[0] < -margin) or \
               (pa[0] > w + margin and pb[0] > w + margin) or \
               (pa[1] < -margin and pb[1] < -margin) or \
               (pa[1] > h + margin and pb[1] > h + margin):
                continue
            is_axis = abs(seg[0][0]) < 1.0 or abs(seg[0][2]) < 1.0
            colour = (0.18, 0.22, 0.25) if is_axis else (0.10, 0.12, 0.14)
            cv2.line(canvas, pa, pb, colour, 2 if is_axis else 1, cv2.LINE_AA)


# ── Legacy alias for backward compatibility ──────────────────────────────────

class PhysicsSkeletonRenderer:
    """Deprecated: use aitviewer SMPL rendering instead."""
    def __init__(self, **_kw):
        log.warning("PhysicsSkeletonRenderer is deprecated — use aitviewer")

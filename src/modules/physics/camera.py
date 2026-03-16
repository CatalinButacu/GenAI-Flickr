
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.shared.constants import DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT


@dataclass(slots=True)
class CameraConfig:
    width: int = DEFAULT_VIDEO_WIDTH     # 640 — see constants.py
    height: int = DEFAULT_VIDEO_HEIGHT   # 480 — see constants.py
    # STANDARD: 60° vertical FOV matches consumer cameras and human perception.
    # PyBullet computeProjectionMatrixFOV expects vertical FOV in degrees.
    fov: float = 60.0
    # ARCH CONSTRAINT: PyBullet near/far clip planes. Objects outside [0.1, 100]
    # metres are culled. 0.1m prevents z-fighting; 100m is generous for scenes.
    near: float = 0.1
    far: float = 100.0
    # DESIGN CHOICE: target slightly above ground (0.3m ≈ knee height) to
    # frame a standing humanoid nicely. [0,0,0] would look at feet.
    target: list[float] = field(default_factory=lambda: [0, 0, 0.3])
    # DESIGN CHOICE: camera placement — 1.5m back, 45° azimuth, -30° elevation.
    # Produces a 3/4 view common in character animation previews.
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0


@dataclass(slots=True)
class FrameData:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


# ── Easing functions ─────────────────────────────────────────────────────────

def easing_smooth(t: float) -> float:
    return t * t * (3 - 2 * t)

def easing_linear(t: float) -> float:
    return t

def easing_ease_in(t: float) -> float:
    return t * t

def easing_ease_out(t: float) -> float:
    return 1 - (1 - t) ** 2

def easing_ease_in_out(t: float) -> float:
    return 3 * t * t - 2 * t * t * t

EASING_FUNCTIONS: dict[str, callable] = {
    "linear":      easing_linear,
    "smooth":      easing_smooth,
    "ease_in":     easing_ease_in,
    "ease_out":    easing_ease_out,
    "ease_in_out": easing_ease_in_out,
}

def apply_easing(t: float, easing: str) -> float:
    """Clamp t to [0,1] and apply the named easing function."""
    t = max(0.0, min(1.0, t))
    return EASING_FUNCTIONS.get(easing, easing_linear)(t)


class CinematicCamera:
    """Time-based camera with orbit / zoom / pitch / pan effects."""

    def __init__(self, target: list[float] = None, distance: float = 2.0,
                 yaw: float = 45.0, pitch: float = -25.0):
        self.base_target = target or [0, 0, 0.5]
        self.base_distance = distance
        self.base_yaw = yaw
        self.base_pitch = pitch

        self.current_target = list(self.base_target)
        self.current_distance = distance
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.effects: list[dict] = []

    # ── effect builders ──────────────────────────────────────────

    def add_orbit(self, start_yaw: float = None, end_yaw: float = None,
                  start_time: float = 0.0, duration: float = 5.0,
                  easing: str = "smooth"):
        self.effects.append({
            "type": "orbit",
            "start_yaw": start_yaw if start_yaw is not None else self.base_yaw,
            "end_yaw": end_yaw if end_yaw is not None else self.base_yaw + 360,
            "start_time": start_time, "duration": duration, "easing": easing,
        })

    def add_zoom(self, start_dist: float = None, end_dist: float = None,
                 start_time: float = 0.0, duration: float = 2.0,
                 easing: str = "smooth"):
        self.effects.append({
            "type": "zoom",
            "start_dist": start_dist if start_dist is not None else self.base_distance,
            "end_dist": end_dist if end_dist is not None else self.base_distance * 0.5,
            "start_time": start_time, "duration": duration, "easing": easing,
        })

    def add_pitch(self, start_pitch: float = None, end_pitch: float = None,
                  start_time: float = 0.0, duration: float = 2.0,
                  easing: str = "smooth"):
        self.effects.append({
            "type": "pitch",
            "start_pitch": start_pitch if start_pitch is not None else self.base_pitch,
            "end_pitch": end_pitch if end_pitch is not None else -45.0,
            "start_time": start_time, "duration": duration, "easing": easing,
        })

    # ── update loop ──────────────────────────────────────────────

    def update(self, time: float):
        for effect in self.effects:
            progress = self.get_progress(time, effect)
            if progress < 0 or progress > 1:
                continue
            t = apply_easing(progress, effect.get("easing", "smooth"))
            self.apply_effect(effect, t)

    def get_progress(self, time: float, effect: dict) -> float:
        start, duration = effect["start_time"], effect["duration"]
        if duration <= 0:
            return 1.0 if time >= start else -1.0
        return (time - start) / duration

    def apply_effect(self, effect: dict, t: float):
        match effect["type"]:
            case "orbit":
                self.current_yaw = self.lerp(effect["start_yaw"], effect["end_yaw"], t)
            case "zoom":
                self.current_distance = self.lerp(effect["start_dist"], effect["end_dist"], t)
            case "pitch":
                self.current_pitch = self.lerp(effect["start_pitch"], effect["end_pitch"], t)
            case "pan":
                s, e = effect["start_target"], effect["end_target"]
                self.current_target = [self.lerp(s[i], e[i], t) for i in range(3)]

    @staticmethod
    def lerp(start: float, end: float, t: float) -> float:
        return start + (end - start) * t

    # ── accessors ────────────────────────────────────────────────

    def get_camera_params(self) -> tuple[float, float, float, list[float]]:
        return (self.current_yaw, self.current_pitch,
                self.current_distance, self.current_target)

    def to_config(self) -> CameraConfig:
        return CameraConfig(
            target=self.current_target, distance=self.current_distance,
            yaw=self.current_yaw, pitch=self.current_pitch,
        )

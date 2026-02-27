"""
#WHERE
    Used by simulator.py (Simulator.run_cinematic), demo_cinematic_camera.py,
    pipeline.py, and examples.

#WHAT
    Cinematic camera system with easing functions and animation effects
    (orbit, zoom, pitch, pan).  Provides CameraConfig for static setup
    and CinematicCamera for time-based animations.

#INPUT
    Camera parameters (yaw, pitch, distance, target) and effect definitions
    with start_time, duration, easing type.

#OUTPUT
    CameraConfig dataclass or (yaw, pitch, distance, target) tuple per frame.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0
    target: List[float] = field(default_factory=lambda: [0, 0, 0.3])
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0


@dataclass
class FrameData:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


class EasingFunctions:
    """Hermite / polynomial easing functions for camera interpolation."""

    @staticmethod
    def smooth(t: float) -> float:
        return t * t * (3 - 2 * t)

    @staticmethod
    def linear(t: float) -> float:
        return t

    @staticmethod
    def ease_in(t: float) -> float:
        return t * t

    @staticmethod
    def ease_out(t: float) -> float:
        return 1 - (1 - t) ** 2

    @staticmethod
    def ease_in_out(t: float) -> float:
        return 3 * t * t - 2 * t * t * t

    FUNCTIONS = {
        "linear": linear.__func__,
        "smooth": smooth.__func__,
        "ease_in": ease_in.__func__,
        "ease_out": ease_out.__func__,
        "ease_in_out": ease_in_out.__func__,
    }

    @staticmethod
    def apply(t: float, easing: str) -> float:
        t = max(0, min(1, t))
        func = EasingFunctions.FUNCTIONS.get(easing, EasingFunctions.linear)
        return func(t)


class CinematicCamera:
    """Time-based camera with orbit / zoom / pitch / pan effects."""

    def __init__(self, target: List[float] = None, distance: float = 2.0,
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

    def add_pan(self, start_target: List[float] = None,
                end_target: List[float] = None,
                start_time: float = 0.0, duration: float = 2.0,
                easing: str = "smooth"):
        self.effects.append({
            "type": "pan",
            "start_target": start_target or list(self.base_target),
            "end_target": end_target or [0, 0, 0.5],
            "start_time": start_time, "duration": duration, "easing": easing,
        })

    # ── update loop ──────────────────────────────────────────────

    def update(self, time: float):
        for effect in self.effects:
            progress = self._get_progress(time, effect)
            if progress < 0 or progress > 1:
                continue
            t = EasingFunctions.apply(progress, effect.get("easing", "smooth"))
            self._apply_effect(effect, t)

    def _get_progress(self, time: float, effect: dict) -> float:
        start, duration = effect["start_time"], effect["duration"]
        if duration <= 0:
            return 1.0 if time >= start else -1.0
        return (time - start) / duration

    def _apply_effect(self, effect: dict, t: float):
        etype = effect["type"]
        if etype == "orbit":
            self.current_yaw = self._lerp(effect["start_yaw"], effect["end_yaw"], t)
        elif etype == "zoom":
            self.current_distance = self._lerp(effect["start_dist"], effect["end_dist"], t)
        elif etype == "pitch":
            self.current_pitch = self._lerp(effect["start_pitch"], effect["end_pitch"], t)
        elif etype == "pan":
            s, e = effect["start_target"], effect["end_target"]
            self.current_target = [self._lerp(s[i], e[i], t) for i in range(3)]

    @staticmethod
    def _lerp(start: float, end: float, t: float) -> float:
        return start + (end - start) * t

    # ── accessors ────────────────────────────────────────────────

    def get_camera_params(self) -> Tuple[float, float, float, List[float]]:
        return (self.current_yaw, self.current_pitch,
                self.current_distance, self.current_target)

    def to_config(self) -> CameraConfig:
        return CameraConfig(
            target=self.current_target, distance=self.current_distance,
            yaw=self.current_yaw, pitch=self.current_pitch,
        )

"""M7: Post-processing renderer.

Applies cinematic effects to raw FrameData from M5 and exports a final MP4.

Effects (all individually toggleable via RenderSettings):
    - Motion blur      — exponential frame accumulation (trailing ghost effect)
    - Depth of field   — foreground/background blur based on the M5 depth buffer
    - Color grading    — saturation, contrast, gamma, optional cinematic crush
    - Vignette         — cosine-falloff corner darkening
    - Film grain       — subtle additive noise for texture

Usage::

    from src.modules.m7_render_engine import RenderEngine, RenderSettings

    settings = RenderSettings(motion_blur=True, dof=True)
    engine = RenderEngine(settings)
    engine.render(frames, "outputs/videos/output.mp4", fps=24)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RenderSettings:
    """Per-render knobs.  All effects default to cinema-safe values."""

    # ── Motion blur ──────────────────────────────────────────────────────────
    motion_blur: bool = True
    motion_blur_alpha: float = 0.6    # weight on current frame; lower = stronger trail

    # ── Depth of field ───────────────────────────────────────────────────────
    dof: bool = True
    dof_focus_depth: Optional[float] = None  # None → auto (median depth per frame)
    dof_max_blur_radius: int = 7             # pixels; must be odd kernel side

    # ── Color grading ─────────────────────────────────────────────────────────
    color_grade: bool = True
    saturation: float = 1.25
    contrast: float = 1.15      
    gamma: float = 0.95         # < 1.0 = brighter midtones
    tint: tuple[float, float, float] = (1.02, 1.0, 0.96)  # slight warm shift

    # ── Vignette ──────────────────────────────────────────────────────────────
    vignette: bool = True
    vignette_strength: float = 0.45   # 0.0 = off, 1.0 = black corners

    # ── Film grain ────────────────────────────────────────────────────────────
    film_grain: bool = False           # off by default — use when M8 is disabled
    grain_sigma: float = 4.0           # standard deviation of Gaussian noise

    # ── Output ────────────────────────────────────────────────────────────────
    output_layout: str = "rgb"         # "rgb" | "horizontal" | "vertical"


class RenderEngine:
    """Post-processes M5 FrameData and writes a cinematic MP4."""

    def __init__(self, settings: Optional[RenderSettings] = None) -> None:
        self._s = settings or RenderSettings()
        self._vignette_cache: Optional[np.ndarray] = None  # memoised per resolution

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """No-op — kept for uniform module interface."""

    def render(self, frames: List, output_path: str, fps: int = 24) -> str:
        """Apply all effects and write MP4.  Returns *output_path*."""
        import imageio

        if not frames:
            log.warning("RenderEngine: received 0 frames — nothing to render")
            return output_path

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        processed = self._process_frames(frames)

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=16)
        for rgb in processed:
            writer.append_data(rgb)
        writer.close()

        log.info("[M7] rendered %d frames → %s", len(processed), output_path)
        return output_path

    # ── Pipeline ─────────────────────────────────────────────────────────────

    def _process_frames(self, frames: List) -> List[np.ndarray]:
        buf: Optional[np.ndarray] = None
        result: List[np.ndarray] = []
        for frame in frames:
            rgb, buf = self._process_one_frame(frame, buf)
            result.append(rgb)
        return result

    def _process_one_frame(
        self, frame, buf: Optional[np.ndarray]
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply all enabled effects to a single frame; returns (uint8 rgb, new buf)."""
        s   = self._s
        rgb = frame.rgb.astype(np.float32) / 255.0
        depth = frame.depth if hasattr(frame, "depth") else None

        if s.motion_blur:
            buf = rgb if buf is None else s.motion_blur_alpha * rgb + (1.0 - s.motion_blur_alpha) * buf
            rgb = buf

        if s.dof and depth is not None:
            rgb = self._apply_dof(rgb, depth)

        if s.color_grade:
            rgb = self._apply_color_grade(rgb)

        if s.vignette:
            rgb = self._apply_vignette(rgb)

        if s.film_grain:
            rgb = self._apply_grain(rgb)

        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), buf

    # ── Effects ──────────────────────────────────────────────────────────────

    def _apply_dof(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Lens-blur based on the depth buffer.

        Strategy: derive per-pixel blur radius from |depth - focus_plane|,
        then blend between the sharp frame and progressively blurred versions.
        Falls back gracefully if cv2 is absent.
        """
        try:
            import cv2
        except ImportError:
            log.debug("[M7] cv2 unavailable — DoF skipped")
            return rgb

        s = self._s
        focus = s.dof_focus_depth if s.dof_focus_depth is not None else float(np.median(depth))

        # Normalised circle-of-confusion  (0 = in focus, 1 = max blur)
        depth_range = float(depth.max() - depth.min()) or 1.0
        coc = np.abs(depth.astype(np.float32) - focus) / depth_range  # (H, W)

        rgb_u8 = (rgb * 255).astype(np.uint8)
        max_r  = s.dof_max_blur_radius | 1   # ensure odd

        # Build blurred version at max radius then blend per-pixel
        blurred = cv2.GaussianBlur(rgb_u8, (max_r, max_r), 0).astype(np.float32) / 255.0
        alpha   = coc[:, :, np.newaxis]    # broadcast over channels
        return (1.0 - alpha) * rgb + alpha * blurred

    def _apply_color_grade(self, rgb: np.ndarray) -> np.ndarray:
        """Saturation boost, contrast, gamma, warm tint — all in float [0,1]."""
        s = self._s

        # ── Saturation (convert to YCbCr-like luminance-preserving scale) ────
        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        lum = lum[:, :, np.newaxis]
        rgb = lum + s.saturation * (rgb - lum)

        # ── Contrast (pivot at 0.5) ───────────────────────────────────────────
        rgb = (rgb - 0.5) * s.contrast + 0.5

        # ── Gamma ─────────────────────────────────────────────────────────────
        rgb = np.power(np.clip(rgb, 0.0, 1.0), s.gamma)

        # ── Warm/cool tint (per-channel multiplier) ────────────────────────────
        rgb = rgb * np.array(s.tint, dtype=np.float32)

        return rgb

    def _apply_vignette(self, rgb: np.ndarray) -> np.ndarray:
        """Cosine-falloff radial mask — darkens corners smoothly."""
        h, w = rgb.shape[:2]
        mask = self._get_vignette_mask(h, w)
        return rgb * mask[:, :, np.newaxis]

    def _get_vignette_mask(self, h: int, w: int) -> np.ndarray:
        """Memoised mask keyed on resolution."""
        if self._vignette_cache is not None and self._vignette_cache.shape == (h, w):
            return self._vignette_cache

        s    = self._s
        cx   = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        cy   = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx, yy = np.meshgrid(cx, cy)
        dist = np.sqrt(xx ** 2 + yy ** 2)                       # 0 at centre, ~1.41 at corners
        dist = np.clip(dist, 0.0, 1.0)
        # Smooth cosine rolloff: 1 at centre, (1-strength) at edge
        mask = 1.0 - s.vignette_strength * (0.5 - 0.5 * np.cos(np.pi * dist))
        self._vignette_cache = mask.astype(np.float32)
        return self._vignette_cache

    def _apply_grain(self, rgb: np.ndarray) -> np.ndarray:
        """Additive Gaussian noise (seed derived from frame content for reproducibility)."""
        seed  = abs(int(rgb.sum() * 1e4)) % (2 ** 31)
        rng   = np.random.default_rng(seed=seed)
        noise = rng.standard_normal(rgb.shape).astype(np.float32) * (self._s.grain_sigma / 255.0)
        return rgb + noise

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import imageio
import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class RenderSettings:
    """Per-render knobs.  All effects default to cinema-safe values."""

    # ── Motion blur ──────────────────────────────────────────────────────────
    motion_blur: bool = True
    # DESIGN CHOICE: alpha=0.6 means 60% current frame + 40% previous frame.
    # Range [0.3, 1.0]. Lower = heavier ghosting. 0.6 is subtle enough for
    # 24fps output without smearing over fine joint movements.
    motion_blur_alpha: float = 0.6

    # ── Depth of field ───────────────────────────────────────────────────────
    dof: bool = True
    dof_focus_depth: float | None = None  # None → auto (median depth per frame)
    # DESIGN CHOICE: 7px max blur at 640×480. Produces gentle background softening
    # without making the subject unrecognisable. Must be odd for GaussianBlur kernel.
    # Scale proportionally if rendering at higher resolution.
    dof_max_blur_radius: int = 7

    # ── Color grading ─────────────────────────────────────────────────────────
    color_grade: bool = True
    # DESIGN CHOICES: cinematic colour correction inspired by film LUTs.
    # saturation=1.25: +25% (silhouettes benefit from colour punch).
    # contrast=1.15: subtle S-curve lift. gamma=0.95: brighten midtones.
    # tint=(1.02,1.0,0.96): warm shift — +2% red, -4% blue (golden-hour feel).
    saturation: float = 1.25
    contrast: float = 1.15      
    gamma: float = 0.95         # < 1.0 = brighter midtones
    tint: tuple[float, float, float] = (1.02, 1.0, 0.96)

    # ── Vignette ──────────────────────────────────────────────────────────────
    vignette: bool = True
    # DESIGN CHOICE: 0.45 is moderate — visible but doesn't obscure corners.
    # Range [0.0, 1.0]. 0.0 = off, 1.0 = pure black corners.
    vignette_strength: float = 0.45

    # ── Film grain ────────────────────────────────────────────────────────────
    film_grain: bool = False           # off by default — use when M8 is disabled
    # DESIGN CHOICE: sigma=4.0 produces barely-visible grain at 640×480.
    # Higher values (8-12) give noticeable film texture.
    grain_sigma: float = 4.0

    # ── Output ────────────────────────────────────────────────────────────────
    output_layout: str = "rgb"         # "rgb" | "horizontal" | "vertical"


class RenderEngine:
    """Post-processes M5 FrameData and writes a cinematic MP4."""

    def __init__(self, settings: RenderSettings | None = None) -> None:
        self.settings = settings or RenderSettings()
        self.vignette_cache: np.ndarray | None = None  # memoised per resolution


    def setup(self) -> None:
        """No-op — kept for uniform module interface."""

    def render(self, frames: list, output_path: str, fps: int = 24) -> str:
        """Apply all effects and write MP4.  Returns *output_path*."""
        if not frames:
            log.warning("RenderEngine: received 0 frames — nothing to render")
            return output_path

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        processed = self.process_frames(frames)

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264",
                                    quality=8, macro_block_size=16)
        for rgb in processed:
            writer.append_data(rgb)
        writer.close()

        log.info("[M6] rendered %d frames → %s", len(processed), output_path)
        return output_path


    def process_frames(self, frames: list) -> list[np.ndarray]:
        buf: np.ndarray | None = None
        result: list[np.ndarray] = []
        for frame in frames:
            rgb, buf = self.process_one_frame(frame, buf)
            result.append(rgb)
        return result

    def process_one_frame(
        self, frame, buf: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply all enabled effects to a single frame; returns (uint8 rgb, new buf)."""
        s   = self.settings
        rgb = frame.rgb.astype(np.float32) / 255.0
        depth = frame.depth if hasattr(frame, "depth") else None

        if s.motion_blur:
            buf = rgb if buf is None else s.motion_blur_alpha * rgb + (1.0 - s.motion_blur_alpha) * buf
            rgb = buf

        if s.dof and depth is not None:
            rgb = self.apply_dof(rgb, depth)

        if s.color_grade:
            rgb = self.apply_color_grade(rgb)

        if s.vignette:
            rgb = self.apply_vignette(rgb)

        if s.film_grain:
            rgb = self.apply_grain(rgb)

        return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8), buf

    # ── Effects ──────────────────────────────────────────────────────────────

    def apply_dof(self, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Lens-blur based on the depth buffer.

        Strategy: derive per-pixel blur radius from |depth - focus_plane|,
        then blend between the sharp frame and progressively blurred versions.
        Falls back gracefully if cv2 is absent.
        """
        s = self.settings
        focus = s.dof_focus_depth if s.dof_focus_depth is not None else float(np.median(depth))

        # Circle of confusion: 0 = in focus, 1 = max blur.
        # Larger distance from focus plane → stronger defocus effect.
        depth_range = float(depth.max() - depth.min()) or 1.0
        coc = np.abs(depth.astype(np.float32) - focus) / depth_range  # (H, W)

        rgb_u8 = (rgb * 255).astype(np.uint8)
        max_r  = s.dof_max_blur_radius | 1   # ensure odd

        # Build blurred version at max radius then blend per-pixel
        blurred = cv2.GaussianBlur(rgb_u8, (max_r, max_r), 0).astype(np.float32) / 255.0
        alpha   = coc[:, :, np.newaxis]    # broadcast over channels
        return (1.0 - alpha) * rgb + alpha * blurred

    def apply_color_grade(self, rgb: np.ndarray) -> np.ndarray:
        """Saturation boost, contrast, gamma, warm tint — all in float [0,1]."""
        s = self.settings

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

    def apply_vignette(self, rgb: np.ndarray) -> np.ndarray:
        """Cosine-falloff radial mask — darkens corners smoothly."""
        h, w = rgb.shape[:2]
        mask = self.get_vignette_mask(h, w)
        return rgb * mask[:, :, np.newaxis]

    def get_vignette_mask(self, h: int, w: int) -> np.ndarray:
        """Memoised mask keyed on resolution."""
        if self.vignette_cache is not None and self.vignette_cache.shape == (h, w):
            return self.vignette_cache

        s    = self.settings
        cx   = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        cy   = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx, yy = np.meshgrid(cx, cy)
        dist = np.sqrt(xx ** 2 + yy ** 2)                       # 0 at centre, ~1.41 at corners
        dist = np.clip(dist, 0.0, 1.0)
        # Smooth cosine rolloff: 1 at centre, (1-strength) at edge
        mask = 1.0 - s.vignette_strength * (0.5 - 0.5 * np.cos(np.pi * dist))
        self.vignette_cache = mask.astype(np.float32)
        return self.vignette_cache

    def apply_grain(self, rgb: np.ndarray) -> np.ndarray:
        """Additive Gaussian noise (seed derived from frame content for reproducibility)."""
        seed  = abs(int(rgb.sum() * 1e4)) % (2 ** 31)
        rng   = np.random.default_rng(seed=seed)
        noise = rng.standard_normal(rgb.shape).astype(np.float32) * (self.settings.grain_sigma / 255.0)
        return rgb + noise

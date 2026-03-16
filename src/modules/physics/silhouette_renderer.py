"""Silhouette renderer: SMPL mesh or analytical fallback → filled body frames.

Used for cinematic video output and ControlNet conditioning images.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .body_model import (
    BodyParams,
    estimate_px_per_m,
    render_body_silhouette,
)
from .physics_renderer import (
    _VirtualCamera,
    _draw_ground_grid,
    _make_background,
    auto_orient_skeleton,
)
from .smpl_body import (
    HAS_SMPL,
    SMPLBody,
    is_smpl_available,
    render_smpl_silhouette,
    smpl_betas_from_body_params,
)

log = logging.getLogger(__name__)

# Resolve SMPL availability once at import time
_SMPL_READY: bool = is_smpl_available()
if _SMPL_READY:
    log.info("SMPL model files found — using mesh-based silhouettes")
else:
    log.debug(
        "SMPL model files not found — using analytical silhouettes.  "
        "Run 'python scripts/setup_smpl.py' for instructions."
    )


class SilhouetteSkeletonRenderer:
    """Render skeletons as filled body silhouettes (SMPL mesh or analytical fallback)."""

    def __init__(
        self,
        # DESIGN CHOICE: 1280×720 = 720p HD, standard for showcase videos.
        # Not architectural — can be any resolution.
        img_w: int = 1280,
        img_h: int = 720,
        # DESIGN CHOICE: 45° yaw = 3/4 view, -25° pitch = slight top-down.
        # Standard character-animation camera angle. Not architectural.
        yaw_deg: float = 45.0,
        pitch_deg: float = -25.0,
        # DESIGN CHOICE: 4.0m distance frames full body in 720p at default FOV.
        distance: float = 4.0,
        target: list[float] | None = None,
        action_label: str = "",
        body_params: BodyParams | None = None,
    ) -> None:
        self.w = img_w
        self.h = img_h
        self.action_label = action_label
        self.body_params = body_params or BodyParams()
        self.cam = _VirtualCamera(
            target=np.array(target or [0.0, 900.0, 0.0]),
            distance=distance,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            img_w=img_w,
            img_h=img_h,
        )
        self._bg = _make_background(img_w, img_h)
        self._prev_xyz: np.ndarray | None = None

        # Initialise SMPL body if model files are available
        self._smpl: SMPLBody | None = None
        if _SMPL_READY:
            try:
                gender = self.body_params.gender if self.body_params.gender in ("male", "female") else "neutral"
                self._smpl = SMPLBody.get_or_create(gender=gender)
                log.info("SilhouetteSkeletonRenderer using SMPL mesh rendering")
            except Exception:
                log.warning("Failed to load SMPL — falling back to analytical model", exc_info=True)
                self._smpl = None

    def update_camera(
        self,
        yaw_deg: float,
        pitch_deg: float,
        distance: float,
        target: list,
    ) -> None:
        self.cam.update(yaw_deg, pitch_deg, distance, target)

    def render_frame(
        self,
        xyz_21: np.ndarray,
        prev_xyz: np.ndarray | None = None,
        body_params: BodyParams | None = None,
    ) -> np.ndarray:
        """Render one frame with a filled body silhouette."""
        params = body_params or self.body_params
        if prev_xyz is None:
            prev_xyz = self._prev_xyz if self._prev_xyz is not None else xyz_21
        self._prev_xyz = xyz_21.copy()

        uvd = self.cam.project(xyz_21)
        px_per_m = estimate_px_per_m(uvd, params.height_m)

        canvas = self._prepare_canvas()
        canvas = self._render_body(canvas, uvd, params, px_per_m)

        if prev_xyz is not None:
            vel = np.linalg.norm(xyz_21 - prev_xyz, axis=-1)
            max_vel = float(vel.max())
            # DESIGN CHOICE: velocity > 30 units/frame triggers motion trail.
            # Tuned for 30fps AMASS data in mm — 30mm/frame ≈ 0.9 m/s.
            if max_vel > 30.0:
                self._draw_motion_trail(canvas, uvd, self.cam.project(prev_xyz),
                                        params, px_per_m, max_vel)

        if self.action_label:
            self._draw_action_label(canvas, self.action_label)
        return canvas

    def _prepare_canvas(self) -> np.ndarray:
        """Create canvas with ground grid drawn."""
        canvas = self._bg.copy()
        canvas_f = canvas.astype(np.float32) / 255.0
        _draw_ground_grid(canvas_f, self.cam, self.w, self.h)
        return (np.clip(canvas_f, 0, 1) * 255).astype(np.uint8)

    def _render_body(self, canvas, uvd, params, px_per_m) -> np.ndarray:
        """Render body silhouette using SMPL mesh or analytical fallback."""
        if self._smpl is not None:
            try:
                betas = smpl_betas_from_body_params(params)
                vertices, faces = self._smpl.get_posed_mesh(uvd, betas=betas)
                return render_smpl_silhouette(
                    vertices, faces, self.cam.project, canvas,
                    fill_color=params.skin_tone, shadow=True, outline_thickness=2,
                )
            except Exception:
                log.debug("SMPL render failed, falling back to analytical",
                          exc_info=True)
        return render_body_silhouette(
            uvd, params, canvas, px_per_m,
            shadow=True, outline_thickness=2,
        )

    def _draw_motion_trail(
        self,
        canvas: np.ndarray,
        uvd_current: np.ndarray,
        uvd_prev: np.ndarray,
        params: BodyParams,
        px_per_m: float,
        max_vel: float,
    ) -> None:
        """Draw a faint trailing silhouette for high-velocity movement."""
        # DESIGN CHOICE: cap ghost opacity at 25% to avoid obscuring the
        # main silhouette. 300.0 divisor = at max_vel=300 ghost is fully 25%.
        alpha = min(0.25, max_vel / 300.0)
        trail_overlay = np.zeros_like(canvas, dtype=np.float32)
        trail_canvas = canvas.copy()

        # Render a ghost at the midpoint between current and previous
        midpoint_uvd = (uvd_current + uvd_prev) * 0.5
        r, g, b = params.skin_tone
        ghost_color = (int(r * 0.5), int(g * 0.5), int(b * 0.5))

        render_body_silhouette(
            midpoint_uvd, params, trail_canvas, px_per_m,
            fill_color=ghost_color,
            outline_thickness=0,
            shadow=False,
        )

        # Blend ghost into canvas
        canvas_f = canvas.astype(np.float32)
        trail_f = trail_canvas.astype(np.float32)
        blended = canvas_f * (1 - alpha) + trail_f * alpha
        np.clip(blended, 0, 255, out=blended)
        canvas[:] = blended.astype(np.uint8)

    @staticmethod
    def _draw_action_label(
        canvas: np.ndarray,
        action_label: str,
    ) -> None:
        """Draw action label overlay on the frame."""
        h, w = canvas.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 1200.0)
        thick = max(1, int(scale * 2))
        text = action_label.upper()
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        x = w - tw - 20
        y = th + 20
        cv2.rectangle(canvas, (x - 10, 2), (w - 2, y + 10), (15, 15, 15), -1)
        cv2.putText(canvas, text, (x, y), font, scale,
                    (220, 200, 180), thick, cv2.LINE_AA)


class SilhouetteProjector:
    """Project 3-D skeleton → filled-body image for ControlNet conditioning."""

    def __init__(
        self,
        # ARCH CONSTRAINT: 512×512 matches SD 1.5 / ControlNet native resolution.
        # ControlNet OpenPose conditioning images must match the diffusion model's
        # latent grid (64×64 → 512×512 pixel space). Using other sizes works but
        # degrades pose accuracy.
        img_w: int = 512,
        img_h: int = 512,
        # DESIGN CHOICE: 15° yaw = near-frontal view for ControlNet conditioning.
        # Slight angle avoids bilateral symmetry ambiguity. Not architectural.
        cam_yaw_deg: float = 15.0,
        body_params: BodyParams | None = None,
    ) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.cam_yaw = np.radians(cam_yaw_deg)
        self.body_params = body_params or BodyParams()

        # Initialise SMPL body if available
        self._smpl: SMPLBody | None = None
        if _SMPL_READY:
            try:
                gender = self.body_params.gender if self.body_params.gender in ("male", "female") else "neutral"
                self._smpl = SMPLBody.get_or_create(gender=gender)
            except Exception:
                self._smpl = None

    def project(self, joints_3d: np.ndarray) -> np.ndarray:
        """Orthographic projection of (21, 3) Y-up mm → (21, 3) pixel coords.

        Returns (21, 3) array of (col, row, depth_norm).
        """
        pts = joints_3d.copy().astype(np.float64)

        # Centre on canvas
        pts[:, 0] -= pts[0, 0]
        pts[:, 2] -= pts[0, 2]

        # Yaw rotation
        cos_y, sin_y = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        x_rot = cos_y * pts[:, 0] + sin_y * pts[:, 2]
        pts[:, 0] = x_rot

        # Screen mapping
        screen_x = pts[:, 0]
        screen_y = pts[:, 1]

        # DESIGN CHOICE: 12% padding prevents limbs from touching image edges.
        # Standard framing margin for body silhouettes. Not architectural.
        padding = 0.12
        x_range = max(screen_x.max() - screen_x.min(), 1.0)
        y_range = max(screen_y.max() - screen_y.min(), 1.0)
        scale = min(
            self.img_w * (1 - 2 * padding) / x_range,
            self.img_h * (1 - 2 * padding) / y_range,
        )

        col = (screen_x - screen_x.min()) * scale + self.img_w * padding
        row = self.img_h - ((screen_y - screen_y.min()) * scale + self.img_h * padding)
        depth = np.zeros(len(pts))  # orthographic — no depth variation

        return np.stack([col, row, depth], axis=-1)

    def render(
        self,
        joints_3d: np.ndarray,
        body_params: BodyParams | None = None,
    ) -> np.ndarray:
        """Project + render silhouette on black background.

        Returns (H, W, 3) uint8 RGB image.
        """
        params = body_params or self.body_params
        canvas = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        if self._smpl is not None:
            try:
                betas = smpl_betas_from_body_params(params)
                vertices, faces = self._smpl.get_posed_mesh(joints_3d, betas=betas)
                # Use the orthographic projector for SMPL vertices
                canvas = render_smpl_silhouette(
                    vertices, faces,
                    self._project_vertices,
                    canvas,
                    fill_color=(220, 210, 200),
                    outline_color=(255, 245, 235),
                    outline_thickness=1,
                    shadow=False,
                )
                return canvas
            except Exception:
                log.debug("SMPL projector render failed, using analytical", exc_info=True)

        # Analytical fallback
        uvd = self.project(joints_3d)
        px_per_m = estimate_px_per_m(uvd, params.height_m)
        canvas = render_body_silhouette(
            uvd, params, canvas, px_per_m,
            fill_color=(220, 210, 200),
            outline_color=(255, 245, 235),
            outline_thickness=1,
            shadow=False,
        )
        return canvas

    def _project_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Orthographic projection for arbitrary (N, 3) vertices (mm, Y-up).

        Returns (N, 3) array of (col, row, depth_norm).
        """
        pts = vertices.copy().astype(np.float64)

        # Centre on pelvis (vertex mean)
        centroid = pts.mean(axis=0)
        pts[:, 0] -= centroid[0]
        pts[:, 2] -= centroid[2]

        # Yaw rotation
        cos_y, sin_y = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        x_rot = cos_y * pts[:, 0] + sin_y * pts[:, 2]
        pts[:, 0] = x_rot

        # Screen mapping
        screen_x = pts[:, 0]
        screen_y = pts[:, 1]

        # DESIGN CHOICE: 12% padding (same as project() above).
        padding = 0.12
        x_range = max(screen_x.max() - screen_x.min(), 1.0)
        y_range = max(screen_y.max() - screen_y.min(), 1.0)
        scale = min(
            self.img_w * (1 - 2 * padding) / x_range,
            self.img_h * (1 - 2 * padding) / y_range,
        )

        col = (screen_x - screen_x.min()) * scale + self.img_w * padding
        row = self.img_h - ((screen_y - screen_y.min()) * scale + self.img_h * padding)
        depth = np.zeros(len(pts))

        return np.stack([col, row, depth], axis=-1)

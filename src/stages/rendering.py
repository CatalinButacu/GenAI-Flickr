"""M6 (Rendering) + M7 (Diffusion) stage — SMPL mesh rendering and video export."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from src.shared.mem_profile import profile_memory
from src.modules.render import RenderEngine, RenderSettings, get_best_renderer
from src.modules.physics import FrameData, CinematicCamera, auto_orient_skeleton
from src.modules.physics.body_model import BodyParams
from src.modules.physics.smpl_body import SMPLBody, is_smpl_available, smpl_betas_from_body_params

log = logging.getLogger(__name__)


class RenderingStage:
    """M6: Render skeleton poses to video frames and export MP4."""

    def __init__(
        self, *,
        fps: int = 24,
        use_render_engine: bool = True,
        use_diffusion: bool = False,
        use_silhouette: bool = True,
        device: str = "cuda",
    ) -> None:
        self.fps = fps
        self.use_render_engine = use_render_engine
        self.use_diffusion = use_diffusion
        self.use_silhouette = use_silhouette
        self.device = device
        self.enhancer: Any = None

    def setup(self) -> None:
        """Optionally initialise AI enhancer for M7 diffusion path."""
        if not self.use_diffusion:
            return
        from src.modules.diffusion import (
            AnimateDiffHumanRenderer, ControlNetHumanRenderer,
        )
        self.enhancer = AnimateDiffHumanRenderer(
            device=self.device, batch_size=4, width=384, height=384,
        )
        if self.enhancer.setup():
            log.info("[M7] AnimateDiff renderer loaded")
            return
        log.warning("[M7] AnimateDiff failed — trying ControlNet")
        self.enhancer = ControlNetHumanRenderer(device=self.device)
        if not self.enhancer.setup():
            raise RuntimeError("ControlNet setup failed")

    # ── public API ──────────────────────────────────────────────────

    def render_frames(
        self,
        skeleton_positions: list,
        cam: CinematicCamera,
        action_label: str = "",
        body_params: BodyParams | None = None,
        smplx_motion: np.ndarray | None = None,
        smplx_betas: np.ndarray | None = None,
    ) -> list[FrameData]:
        """Render skeleton poses to RGB frames (M6 or M7 path).

        When *smplx_motion* is provided, renders via aitviewer SMPLSequence
        for publication-quality output identical to render_interx_demo.py.
        """
        use_diffusion = (
            self.use_diffusion and self.enhancer is not None
            and self.enhancer.is_ready
        )
        if use_diffusion:
            raw = self._render_diffusion(skeleton_positions, body_params)
        elif smplx_motion is not None:
            raw = self._render_smplx(smplx_motion, smplx_betas)
        else:
            raw = self._render_mesh(skeleton_positions, cam, body_params)

        # Wrap as FrameData
        h, w = raw[0].shape[:2]
        dummy_depth = np.zeros((h, w), dtype=np.uint8)
        dummy_seg = np.zeros((h, w), dtype=np.int32)
        return [
            FrameData(timestamp=i / self.fps, rgb=f,
                      depth=dummy_depth, segmentation=dummy_seg)
            for i, f in enumerate(raw)
        ]

    @profile_memory
    def export_video(
        self, sim, frames: list[FrameData], output_name: str,
        output_dir: str = "outputs",
    ) -> str:
        """Encode frames to MP4 with optional post-processing effects."""
        videos_dir = os.path.join(output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        path = os.path.join(videos_dir, f"{output_name}.mp4")

        if self.use_render_engine:
            engine = RenderEngine(RenderSettings(
                motion_blur=True, dof=True, color_grade=True,
                vignette=True, film_grain=False, output_layout="rgb",
            ))
            engine.render(frames, path, fps=self.fps)
        else:
            sim.create_video(frames, path, fps=self.fps, layout="rgb")

        log.info("[M6] video saved -> %s (%d frames @ %dfps)",
                 path, len(frames), self.fps)
        return path

    # ── SMPL-X native rendering (matches render_interx_demo.py) ────

    def _render_smplx(self, smplx_motion, smplx_betas):
        """Render from raw (T, 168) SMPL-X params via SMPLSequence."""
        renderer = get_best_renderer(img_w=1280, img_h=720)
        try:
            return renderer.render_smplx_params(
                smplx_motion,
                betas=smplx_betas,
                fps=self.fps,
            )
        finally:
            renderer.close()

    # ── mesh rendering (M6) ─────────────────────────────────────────

    def _render_mesh(self, skeleton_positions, cam, body_params):
        """Render via aitviewer SMPL meshes."""
        oriented = auto_orient_skeleton(skeleton_positions)
        self._update_camera(oriented[0], cam)

        smpl, betas = self._load_smpl(body_params)
        skin = self._skin_color(body_params)

        renderer = get_best_renderer(img_w=1280, img_h=720)
        try:
            verts_seq, faces = self._build_vertices(oriented, smpl, betas)
            cam.update(0.0)
            yaw, pitch, dist, target = cam.get_camera_params()
            return renderer.render_sequence(
                verts_seq, faces,
                cam_target=np.array(target), cam_distance=dist,
                cam_yaw_deg=yaw, cam_pitch_deg=pitch,
                skin_color=skin, fps=self.fps,
            )
        finally:
            renderer.close()

    def _build_vertices(self, skeleton_positions, smpl, betas):
        verts_seq, faces = [], None
        for xyz in skeleton_positions:
            if smpl is not None:
                try:
                    v, f = smpl.get_posed_mesh(xyz, betas=betas)
                    v = v / 1000.0
                except Exception:
                    v, f = _skeleton_to_mesh(xyz)
            else:
                v, f = _skeleton_to_mesh(xyz)
            if faces is None:
                faces = f
            verts_seq.append(v)
        return verts_seq, faces

    @staticmethod
    def _load_smpl(body_params):
        if not is_smpl_available():
            return None, None
        try:
            gender = "neutral"
            if body_params and body_params.gender in ("male", "female"):
                gender = body_params.gender
            smpl = SMPLBody.get_or_create(gender=gender)
            betas = smpl_betas_from_body_params(body_params) if body_params else None
            return smpl, betas
        except Exception:
            log.warning("[M6] SMPL load failed — using fallback", exc_info=True)
            return None, None

    @staticmethod
    def _skin_color(body_params):
        if body_params and body_params.skin_tone:
            r, g, b = body_params.skin_tone
            return (r / 255, g / 255, b / 255)
        return None

    @staticmethod
    def _update_camera(first_frame, cam):
        y = float((first_frame[0, 1] + first_frame[3, 1]) / 2) / 1000.0
        z = float(np.median(first_frame[:, 2])) / 1000.0
        x = float(np.median(first_frame[:, 0])) / 1000.0
        core = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]
        h = float(first_frame[core, 1].max() - first_frame[core, 1].min()) / 1000.0
        d = max(3.5, h * 1.3)
        cam.base_target = cam.current_target = [x, y, z]
        cam.base_distance = cam.current_distance = d

    # ── diffusion rendering (M7) ────────────────────────────────────

    def _render_diffusion(self, skeleton_positions, body_params):
        """Render via AI diffusion (ControlNet / AnimateDiff)."""
        projector = self._select_projector(body_params)
        positions = skeleton_positions[:32]  # cap for VRAM

        from src.modules.diffusion import AnimateDiffHumanRenderer
        if isinstance(self.enhancer, AnimateDiffHumanRenderer):
            imgs = [projector.render(xyz) for xyz in positions]
            return self.enhancer.render_sequence(imgs)

        # Per-frame ControlNet
        indices = _keyframe_indices(len(positions), interval=4)
        imgs = [projector.render(positions[i]) for i in indices]
        kf_rgb = self.enhancer.render_sequence(imgs)
        return _interpolate_keyframes(kf_rgb, indices, len(positions))

    def _select_projector(self, body_params):
        if self.use_silhouette and body_params is not None:
            from src.modules.physics.silhouette_renderer import SilhouetteProjector
            return SilhouetteProjector(
                img_w=512, img_h=512, cam_yaw_deg=15.0,
                body_params=body_params,
            )
        from src.modules.diffusion import SkeletonProjector
        return SkeletonProjector(
            img_w=512, img_h=512, cam_yaw_deg=15.0,
            joint_radius=5, bone_thickness=3,
        )


# ── module-level helpers ───────────────────────────────────────────────

def _skeleton_to_mesh(xyz_21: np.ndarray):
    """Fallback: sphere-per-joint mesh when SMPL unavailable."""
    import trimesh
    meshes = [
        trimesh.creation.icosphere(subdivisions=2, radius=25.0).apply_translation(xyz_21[j])
        for j in range(xyz_21.shape[0])
    ]
    combined = trimesh.util.concatenate(meshes)
    return combined.vertices / 1000.0, combined.faces


def _keyframe_indices(total: int, interval: int) -> list[int]:
    if total <= interval * 2:
        return list(range(total))
    indices = list(range(0, total, interval))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return indices


def _interpolate_keyframes(kf_rgb, kf_indices, total):
    if len(kf_indices) == total:
        return kf_rgb
    frames: list[np.ndarray] = [np.empty(0)] * total
    for k, idx in enumerate(kf_indices):
        frames[idx] = kf_rgb[k]
    for seg in range(len(kf_indices) - 1):
        s, e = kf_indices[seg], kf_indices[seg + 1]
        sf, ef = kf_rgb[seg].astype(np.float32), kf_rgb[seg + 1].astype(np.float32)
        gap = e - s
        for j in range(1, gap):
            alpha = j / gap
            frames[s + j] = ((1.0 - alpha) * sf + alpha * ef).astype(np.uint8)
    return frames

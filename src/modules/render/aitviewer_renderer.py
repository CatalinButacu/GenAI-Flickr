"""aitviewer-based headless SMPL mesh renderer.

Replaces both the pyrender ``MeshRenderer`` and Open3D ``Open3DRenderer``
with aitviewer's GPU-accelerated headless backend.  Produces
publication-quality frames with PBR lighting, shadows, and a neutral
studio backdrop — identical quality to the Inter-X demo renders.

The renderer works in two modes:

* **Per-frame** — ``render_frame()`` matches the old renderer API
  (vertices + faces per call, returns (H, W, 3) uint8).
* **Batch** — ``render_sequence()`` loads the entire vertex sequence
  into an aitviewer ``Meshes`` object and renders all frames in one
  shot (much faster due to GPU batching).
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np

log = logging.getLogger(__name__)

# ── aitviewer configuration (must happen before other aitviewer imports) ──────
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SMPLX_DIR = os.path.join(_ROOT, "data", "arctic", "unpack", "models")

from aitviewer.configuration import CONFIG as C

C.update_conf({
    "smplx_models": _SMPLX_DIR,
    "window_width": 1280,
    "window_height": 720,
})

from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.meshes import Meshes

# ── Visual constants ─────────────────────────────────────────────────────────

# Neutral studio background — soft blue-grey
_BG_COLOR = [0.85, 0.87, 0.90, 1.0]

# Warm skin-tone default (same as old renderers)
_DEFAULT_SKIN = (0.72, 0.62, 0.55, 1.0)

# Floor — single-colour ground plane (no checkerboard)
_FLOOR_COLOR_A = (0.82, 0.83, 0.84, 1.0)
_FLOOR_COLOR_B = (0.80, 0.81, 0.82, 1.0)  # near-identical → visually solid


class AitviewerRenderer:
    """Headless GPU renderer backed by aitviewer.

    Parameters
    ----------
    img_w, img_h : int
        Output resolution (default 1280x720).
    """

    def __init__(self, img_w: int = 1280, img_h: int = 720) -> None:
        self.w = img_w
        self.h = img_h
        C.update_conf({"window_width": img_w, "window_height": img_h})
        self._viewer: HeadlessRenderer | None = None
        self._mesh_node = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_viewer(self) -> HeadlessRenderer:
        """Lazily create the headless viewer (expensive — done once)."""
        if self._viewer is None:
            self._viewer = HeadlessRenderer()
            s = self._viewer.scene

            # Neutral studio background
            s.background_color = _BG_COLOR

            # Replace default checkerboard with near-solid floor
            if s.floor is not None:
                s.remove(s.floor)
            from aitviewer.renderables.plane import ChessboardPlane
            s.floor = ChessboardPlane(
                100.0, 200,
                _FLOOR_COLOR_A, _FLOOR_COLOR_B,
                "xz",
                name="Floor",
            )
            s.floor.material.diffuse = 0.1
            s.add(s.floor)

        return self._viewer

    def close(self) -> None:
        """Release GPU resources."""
        self._viewer = None

    def __del__(self) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API — per-frame (compatible with old renderer signature)
    # ------------------------------------------------------------------

    def render_frame(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        cam_target: np.ndarray | None = None,
        cam_distance: float = 3.5,
        cam_yaw_deg: float = 30.0,
        cam_pitch_deg: float = -20.0,
        skin_color: tuple[float, float, float] | None = None,
        **_kwargs,
    ) -> np.ndarray:
        """Render one frame of a mesh. Returns (H, W, 3) uint8 RGB."""
        v = self._ensure_viewer()
        s = v.scene

        # Skin colour
        if skin_color is not None:
            color = (*skin_color, 1.0)
        else:
            color = _DEFAULT_SKIN

        verts = np.asarray(vertices, dtype=np.float32).reshape(1, -1, 3)
        faces_arr = np.asarray(faces, dtype=np.int32)
        mesh = Meshes(verts, faces_arr, color=color, name="Body")
        s.add(mesh)

        # Camera
        if cam_target is None:
            cam_target = vertices.mean(axis=0)
        cam_pos = self._orbit_position(cam_target, cam_distance,
                                       cam_yaw_deg, cam_pitch_deg)
        s.camera.position = np.array(cam_pos, dtype=np.float64)
        s.camera.target = np.array(cam_target, dtype=np.float64)

        # Render single frame
        pil_img = v.get_frame()
        rgb = np.array(pil_img)[:, :, :3]  # drop alpha if present
        return rgb

    # ------------------------------------------------------------------
    # Public API — batch sequence (preferred for full motions)
    # ------------------------------------------------------------------

    def render_sequence(
        self,
        vertices_seq: list[np.ndarray] | np.ndarray,
        faces: np.ndarray,
        *,
        cam_target: np.ndarray | None = None,
        cam_distance: float = 3.5,
        cam_yaw_deg: float = 30.0,
        cam_pitch_deg: float = -20.0,
        skin_color: tuple[float, float, float] | None = None,
        fps: int = 24,
    ) -> list[np.ndarray]:
        """Render a full motion sequence using aitviewer's batch rendering.

        Parameters
        ----------
        vertices_seq : list of (V, 3) or (N, V, 3) — metres, Y-up.
        faces : (F, 3)
        cam_target, cam_distance, cam_yaw_deg, cam_pitch_deg : camera params.
        skin_color : (r, g, b) 0-1.
        fps : playback FPS.

        Returns
        -------
        list of (H, W, 3) uint8 RGB frames.
        """
        if isinstance(vertices_seq, list):
            verts_batch = np.stack(vertices_seq, axis=0).astype(np.float32)
        else:
            verts_batch = np.asarray(vertices_seq, dtype=np.float32)

        n_frames = verts_batch.shape[0]
        faces_arr = np.asarray(faces, dtype=np.int32)

        if skin_color is not None:
            color = (*skin_color, 1.0)
        else:
            color = _DEFAULT_SKIN

        v = self._ensure_viewer()
        s = v.scene

        # Add animated mesh
        mesh = Meshes(verts_batch, faces_arr, color=color, name="Body")
        s.add(mesh)

        # Camera setup
        if cam_target is None:
            cam_target = verts_batch[0].mean(axis=0)
        cam_pos = self._orbit_position(cam_target, cam_distance,
                                       cam_yaw_deg, cam_pitch_deg)
        s.camera.position = np.array(cam_pos, dtype=np.float64)
        s.camera.target = np.array(cam_target, dtype=np.float64)

        v.scene.fps = fps
        v.playback_fps = fps

        # Render all frames to a temp dir, then read back as numpy
        with tempfile.TemporaryDirectory(prefix="aitv_render_") as tmp:
            v.save_video(frame_dir=tmp, video_dir=None, output_fps=fps)

            # aitviewer saves frames in a numbered subdirectory (e.g. 0000/)
            sub = tmp
            subdirs = [os.path.join(tmp, d) for d in os.listdir(tmp)
                       if os.path.isdir(os.path.join(tmp, d))]
            if subdirs:
                sub = subdirs[0]

            import cv2
            pngs = sorted(f for f in os.listdir(sub) if f.endswith(".png"))
            frames = []
            for fname in pngs:
                img = cv2.imread(os.path.join(sub, fname))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        log.info("[aitviewer] rendered %d / %d frames", len(frames), n_frames)
        return frames

    # ------------------------------------------------------------------
    # Public API — render from raw SMPL-X parameters (SMPLSequence path)
    # ------------------------------------------------------------------

    def render_smplx_params(
        self,
        smplx_motion: np.ndarray,
        *,
        betas: np.ndarray | None = None,
        fps: int = 30,
        color: tuple[float, ...] = (0.11, 0.53, 0.8, 1.0),
    ) -> list[np.ndarray]:
        """Render from raw (T, 168) SMPL-X parameters via aitviewer SMPLSequence.

        This produces identical output to render_interx_demo.py — proper
        LBS skinning via aitviewer's SMPLLayer, no IK round-trip.

        Parameters
        ----------
        smplx_motion : (T, 168) — root_orient(3)+trans(3)+body(63)+hands(90)+jaw(3)+eye(6)
        betas : (16,) shape coefficients (optional)
        fps : playback framerate
        color : RGBA body colour

        Returns
        -------
        list of (H, W, 3) uint8 RGB frames.
        """
        from aitviewer.models.smpl import SMPLLayer
        from aitviewer.renderables.smpl import SMPLSequence

        T = smplx_motion.shape[0]
        root_orient = smplx_motion[:, :3]
        trans       = smplx_motion[:, 3:6]
        pose_body   = smplx_motion[:, 6:69]
        pose_hand   = smplx_motion[:, 69:159]
        # jaw(3) + eye(6) → not used by SMPLLayer, skip

        smpl_layer = SMPLLayer(model_type="smplx", gender="neutral")

        if betas is None:
            betas = np.zeros(10, dtype=np.float32)
        betas_seq = np.tile(betas[:10].astype(np.float32), (T, 1))

        seq = SMPLSequence(
            poses_body=pose_body.astype(np.float32),
            smpl_layer=smpl_layer,
            poses_root=root_orient.astype(np.float32),
            trans=trans.astype(np.float32),
            betas=betas_seq,
            poses_left_hand=pose_hand[:, :45].astype(np.float32),
            poses_right_hand=pose_hand[:, 45:].astype(np.float32),
            color=color,
        )

        v = self._ensure_viewer()
        s = v.scene
        s.add(seq)

        # Camera — same as render_interx_demo.py
        s.camera.position = np.array([0.0, 1.5, 4.5])
        s.camera.target   = np.array([0.0, 1.0, 0.0])
        s.fps = fps
        v.playback_fps = fps

        with tempfile.TemporaryDirectory(prefix="aitv_smplx_") as tmp:
            v.save_video(frame_dir=tmp, video_dir=None, output_fps=fps)

            sub = tmp
            subdirs = [os.path.join(tmp, d) for d in os.listdir(tmp)
                       if os.path.isdir(os.path.join(tmp, d))]
            if subdirs:
                sub = subdirs[0]

            import cv2
            pngs = sorted(f for f in os.listdir(sub) if f.endswith(".png"))
            frames = []
            for fname in pngs:
                img = cv2.imread(os.path.join(sub, fname))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        log.info("[aitviewer] rendered %d SMPL-X frames via SMPLSequence", len(frames))
        return frames

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _orbit_position(
        target: np.ndarray,
        distance: float,
        yaw_deg: float,
        pitch_deg: float,
    ) -> list[float]:
        """Compute camera eye position on a spherical orbit around *target*."""
        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        return [
            float(target[0] + distance * cp * sy),
            float(target[1] - distance * sp),
            float(target[2] + distance * cp * cy),
        ]

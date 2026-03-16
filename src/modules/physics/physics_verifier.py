from __future__ import annotations

import logging
from dataclasses import dataclass, field

import os
import urllib.request

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None  # type: ignore[assignment]
    HAS_MEDIAPIPE = False

log = logging.getLogger(__name__)

# ── 21-joint humanoid ↔ MediaPipe joint mapping ────────────────────────
# 21-joint humanoid has 21 joints; MediaPipe Pose has 33 landmarks.
# We map the overlapping joints for comparison.
#
# Humanoid index → MediaPipe landmark index
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_HUMANOID_TO_MEDIAPIPE: dict[int, int] = {
    0:  24,  # root/pelvis → left_hip (approx midpoint; see note below)
    3:  0,   # neck → nose (closest head-region landmark)
    4:  0,   # head → nose
    5:  11,  # left_shoulder → left_shoulder
    6:  13,  # left_elbow → left_elbow
    7:  15,  # left_wrist → left_wrist
    8:  12,  # right_shoulder → right_shoulder
    9:  14,  # right_elbow → right_elbow
    10: 16,  # right_wrist → right_wrist
    11: 23,  # left_hip → left_hip
    12: 25,  # left_knee → left_knee
    13: 27,  # left_ankle → left_ankle
    16: 24,  # right_hip → right_hip
    17: 26,  # right_knee → right_knee
    18: 28,  # right_ankle → right_ankle
}

# For pelvis (joint 0), use midpoint of MediaPipe hips 23+24
_PELVIS_KIT = 0
_PELVIS_MP_LEFT = 23
_PELVIS_MP_RIGHT = 24


@dataclass(slots=True)
class VerificationResult:
    """Result of verifying one frame's physics adherence."""
    frame_index: int
    joint_errors_px: dict[int, float] = field(default_factory=dict)
    detected_joints: int = 0
    total_joints: int = len(_HUMANOID_TO_MEDIAPIPE)
    adherence_score: float = 0.0  # 0-100%
    mean_error_px: float = 0.0
    max_error_px: float = 0.0
    detection_succeeded: bool = False
    # Alignment-corrected metrics (translation + uniform scale)
    aligned_mean_error_px: float = 0.0
    aligned_adherence_score: float = 0.0
    offset_px: float = 0.0       # centroid displacement
    scale_ratio: float = 1.0     # detected / physics span


class PhysicsAdherenceVerifier:
    """Verify that AI-rendered frames follow the physics-verified skeleton.

    Uses MediaPipe Pose to detect the human skeleton in each RGB frame,
    then computes per-joint error against the conditioning skeleton.

    Parameters
    ----------
    threshold_px : float
        Maximum acceptable joint displacement in pixels.  Joints within
        this threshold count as "adherent".  Default 30 px at 512×512
        (roughly 6% of image width).
    img_size : tuple
        Expected image size (width, height) for normalisation.
    """

    def __init__(
        self,
        # DESIGN CHOICE: 30px threshold at 512×512 ≈ 5.9% of image width.
        # Generous enough to absorb MediaPipe jitter (±10px) while still
        # catching gross skeleton–render misalignment.  Not architectural.
        threshold_px: float = 30.0,
        # DESIGN CHOICE: 512×512 matches ControlNet conditioning resolution.
        # If rendering at different size, scale threshold_px proportionally.
        img_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.threshold_px = threshold_px
        self.img_w, self.img_h = img_size
        self._pose = None

    def setup(self) -> bool:
        """Initialise MediaPipe PoseLandmarker (task API, v0.10+)."""
        if not HAS_MEDIAPIPE:
            log.warning("mediapipe not installed — physics verifier disabled")
            return False
        model_path = os.path.join("tmp", "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            os.makedirs("tmp", exist_ok=True)
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/"
                "latest/pose_landmarker_lite.task"
            )
            log.info("Downloading PoseLandmarker model…")
            urllib.request.urlretrieve(url, model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=model_path,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(
            options,
        )
        log.info("PhysicsAdherenceVerifier ready (MediaPipe PoseLandmarker)")
        return True

    def get_joint_pixel(
        self,
        kit_idx: int,
        mp_idx: int,
        landmarks: list,
    ) -> tuple[float, float] | None:
        """Return (x, y) pixel coords for a single joint, or None if
        the landmark is not visible enough."""
        lm = landmarks[mp_idx]
        vis = getattr(lm, "visibility", 1.0)
        # DESIGN CHOICE: visibility < 0.3 → unreliable detection.
        # MediaPipe visibility is [0,1]; 0.3 is a common threshold.
        if vis < 0.3:
            return None

        mp_x = lm.x * self.img_w
        mp_y = lm.y * self.img_h

        # Pelvis: midpoint of left/right hips
        if kit_idx == _PELVIS_KIT:
            lm_l = landmarks[_PELVIS_MP_LEFT]
            lm_r = landmarks[_PELVIS_MP_RIGHT]
            vis_l = getattr(lm_l, "visibility", 1.0)
            vis_r = getattr(lm_r, "visibility", 1.0)
            if vis_l > 0.3 and vis_r > 0.3:
                mp_x = (lm_l.x + lm_r.x) / 2 * self.img_w
                mp_y = (lm_l.y + lm_r.y) / 2 * self.img_h
            else:
                return None

        return (mp_x, mp_y)

    @staticmethod
    def procrustes_metrics(
        phys_pts: list[tuple[float, float]],
        mp_pts: list[tuple[float, float]],
        result: VerificationResult,
        threshold: float,
    ) -> None:
        """Compute alignment-corrected (translate + scale) metrics."""
        pa = np.array(phys_pts)
        ma = np.array(mp_pts)
        pc, mc = pa.mean(axis=0), ma.mean(axis=0)
        result.offset_px = float(np.linalg.norm(mc - pc))

        pa_c = pa - pc
        ma_c = ma - mc
        p_span = max(np.linalg.norm(pa.max(0) - pa.min(0)), 1.0)
        m_span = max(np.linalg.norm(ma.max(0) - ma.min(0)), 1.0)
        result.scale_ratio = float(m_span / p_span)

        ma_scaled = ma_c * (p_span / m_span) if m_span > 1.0 else ma_c
        aligned_errors = np.linalg.norm(ma_scaled - pa_c, axis=1)
        result.aligned_mean_error_px = float(aligned_errors.mean())
        aligned_ok = sum(1 for e in aligned_errors if e <= threshold)
        result.aligned_adherence_score = 100.0 * aligned_ok / len(aligned_errors)

    def verify_frame(
        self,
        rendered_rgb: np.ndarray,
        physics_uv: np.ndarray,
        frame_index: int = 0,
    ) -> VerificationResult:
        """Compare a rendered frame against its physics-verified skeleton.

        Parameters
        ----------
        rendered_rgb : ndarray (H, W, 3) uint8
            The photorealistic frame produced by ControlNet/AnimateDiff.
        physics_uv : ndarray (21, 2) int32
            The 2-D pixel coordinates of the physics-verified skeleton
            (as returned by ``SkeletonProjector.project()``).
        frame_index : int
            Frame number (for logging).

        Returns
        -------
        VerificationResult
        """
        result = VerificationResult(frame_index=frame_index)

        landmarks = self._detect_landmarks(rendered_rgb, frame_index)
        if landmarks is None:
            return result
        result.detection_succeeded = True

        errors, phys_pts, mp_pts = self._match_joints(
            physics_uv, landmarks, result,
        )

        result.detected_joints = len(errors)
        if errors:
            result.mean_error_px = float(np.mean(errors))
            result.max_error_px = float(np.max(errors))
            adherent = sum(1 for e in errors if e <= self.threshold_px)
            result.adherence_score = 100.0 * adherent / len(errors)

        if len(phys_pts) >= 3:
            self.procrustes_metrics(phys_pts, mp_pts, result, self.threshold_px)
        return result

    def _detect_landmarks(self, rendered_rgb, frame_index):
        """Run MediaPipe PoseLandmarker, return landmarks or None."""
        if self._pose is None:
            log.warning("MediaPipe PoseLandmarker not initialised")
            return None
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rendered_rgb),
        )
        mp_result = self._pose.detect(mp_image)
        if not mp_result.pose_landmarks:
            log.warning("Frame %d: no pose detected by MediaPipe", frame_index)
            return None
        return mp_result.pose_landmarks[0]

    def _match_joints(self, physics_uv, landmarks, result):
        """Collect per-joint errors between physics and detected pose."""
        errors: list[float] = []
        phys_pts: list[tuple[float, float]] = []
        mp_pts: list[tuple[float, float]] = []
        for kit_idx, mp_idx in _HUMANOID_TO_MEDIAPIPE.items():
            if (mp_xy := self.get_joint_pixel(kit_idx, mp_idx, landmarks)) is None:
                continue
            phys_x = float(physics_uv[kit_idx, 0])
            phys_y = float(physics_uv[kit_idx, 1])
            err = float(np.sqrt((mp_xy[0] - phys_x) ** 2 + (mp_xy[1] - phys_y) ** 2))
            result.joint_errors_px[kit_idx] = err
            errors.append(err)
            phys_pts.append((phys_x, phys_y))
            mp_pts.append(mp_xy)
        return errors, phys_pts, mp_pts

    def verify_sequence(
        self,
        rendered_frames: list[np.ndarray],
        physics_uvs: list[np.ndarray],
    ) -> list[VerificationResult]:
        """Verify all frames in a sequence."""
        results = []
        for i, (frame, uv) in enumerate(zip(rendered_frames, physics_uvs)):
            r = self.verify_frame(frame, uv, frame_index=i)
            results.append(r)
            if (i + 1) % 5 == 0 or i == len(rendered_frames) - 1:
                log.info("Verified %d/%d frames (last: %.1f%% adherence, %.1fpx mean error)",
                         i + 1, len(rendered_frames),
                         r.adherence_score, r.mean_error_px)
        return results

    def summary(self, results: list[VerificationResult]) -> dict[str, float]:
        """Aggregate verification results into summary statistics."""
        if not (detected := [r for r in results if r.detection_succeeded]):
            return {"detection_rate": 0.0}

        return {
            "detection_rate": 100.0 * len(detected) / len(results),
            "mean_adherence": float(np.mean([r.adherence_score for r in detected])),
            "mean_error_px": float(np.mean([r.mean_error_px for r in detected])),
            "max_error_px": float(np.max([r.max_error_px for r in detected])),
            "mean_detected_joints": float(np.mean([r.detected_joints for r in detected])),
            # DESIGN CHOICE: 80% adherence = "acceptable" quality gate.
            # Below 80% the frame likely has a visible limb misplacement.
            "frames_above_80pct": sum(1 for r in detected if r.adherence_score >= 80.0),
            "total_frames": len(results),
        }

    def cleanup(self) -> None:
        """Release MediaPipe resources."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

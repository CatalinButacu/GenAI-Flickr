"""
Physics adherence test — verifies ControlNet output follows the physics skeleton.

Uses the already-generated photorealistic frame from test_controlnet_1frame.py
and compares MediaPipe's detected pose against the physics-verified skeleton.

Run:  python examples/test_physics_adherence.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _load_inputs(out_dir: str) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Load the photorealistic image and physics skeleton UV coordinates."""
    photo_path = os.path.join(out_dir, "photorealistic_output.png")
    if not os.path.exists(photo_path):
        log.error("No photorealistic output found — run test_controlnet_1frame.py first")
        return None

    photo_bgr = cv2.imread(photo_path)
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    h, w = photo_rgb.shape[:2]
    log.info("Loaded %s (%dx%d)", photo_path, w, h)

    uv_path = os.path.join(out_dir, "physics_uv.npy")
    if os.path.exists(uv_path):
        physics_uv = np.load(uv_path)
        log.info("Loaded saved physics UV from %s  (shape %s)", uv_path, physics_uv.shape)
    else:
        log.warning("No saved physics_uv.npy — regenerating (may not match conditioning!)")
        from src.modules.m4_motion_generator import MotionGenerator
        from src.modules.m8_ai_enhancer import SkeletonProjector

        mg = MotionGenerator(use_retrieval=True, use_ssm=False)
        clip = mg.generate("walk forward", num_frames=60)
        if clip is None or clip.raw_joints is None:
            log.error("No raw_joints")
            return None

        mid_idx = len(clip.raw_joints) // 2
        joints_3d = clip.raw_joints[mid_idx]
        projector = SkeletonProjector(img_w=w, img_h=h, cam_yaw_deg=15.0)
        physics_uv = projector.project(joints_3d)

    log.info("Physics skeleton: %d joints projected to 2D", len(physics_uv))
    return photo_rgb, physics_uv, w, h


def _print_report(result, w: int, h: int) -> None:
    """Print the adherence verification results table."""
    print(f"\n{'━' * 55}")
    print("  PHYSICS ADHERENCE VERIFICATION")
    print(f"{'━' * 55}")
    print(f"  Pose detected:     {'YES' if result.detection_succeeded else 'NO'}")
    print(f"  Joints detected:   {result.detected_joints}/{result.total_joints}")
    print()
    print("  -- Raw (pixel-space) --")
    print(f"  Adherence score:   {result.adherence_score:.1f}%")
    print(f"  Mean joint error:  {result.mean_error_px:.1f} px")
    print(f"  Max joint error:   {result.max_error_px:.1f} px")
    print()
    print("  -- Alignment-corrected (translate + scale) --")
    print(f"  Centroid offset:   {result.offset_px:.1f} px")
    print(f"  Scale ratio:       {result.scale_ratio:.2f}")
    print(f"  Aligned adherence: {result.aligned_adherence_score:.1f}%")
    print(f"  Aligned mean err:  {result.aligned_mean_error_px:.1f} px")
    print(f"  Threshold:         {30.0} px  (at {w}\u00d7{h})")
    print()

    if result.joint_errors_px:
        joint_names = [
            "root", "spine1", "spine2", "neck", "head",
            "L_shoulder", "L_elbow", "L_wrist",
            "R_shoulder", "R_elbow", "R_wrist",
            "L_hip", "L_knee", "L_ankle", "L_toe_base", "L_toe_tip",
            "R_hip", "R_knee", "R_ankle", "R_toe_base", "R_toe_tip",
        ]
        print("  Per-joint errors:")
        for jid, err in sorted(result.joint_errors_px.items()):
            name = joint_names[jid] if jid < len(joint_names) else f"joint_{jid}"
            status = "\u2713" if err <= 30.0 else "\u2717"
            print(f"    {status} {name:15s} {err:6.1f} px")

    print(f"{'━' * 55}")


def _draw_overlay(photo_rgb: np.ndarray, physics_uv: np.ndarray,
                  result, w: int, h: int, out_dir: str) -> None:
    """Draw both skeletons (physics + MediaPipe) on the rendered frame."""
    photo_bgr = cv2.cvtColor(photo_rgb, cv2.COLOR_RGB2BGR)
    vis = photo_bgr.copy()

    # Physics skeleton in GREEN
    from src.modules.m8_ai_enhancer.controlnet_human import BONES
    for j_a, j_b in BONES:
        pt_a = (int(physics_uv[j_a, 0]), int(physics_uv[j_a, 1]))
        pt_b = (int(physics_uv[j_b, 0]), int(physics_uv[j_b, 1]))
        cv2.line(vis, pt_a, pt_b, (0, 255, 0), 2, cv2.LINE_AA)
    for col_px, row_px in physics_uv:
        cv2.circle(vis, (int(col_px), int(row_px)), 4, (0, 255, 0), -1)

    # MediaPipe detected skeleton in RED
    _draw_mediapipe_skeleton(vis, photo_rgb, w, h)

    # Legend
    cv2.putText(vis, "GREEN = Physics skeleton (ground truth)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(vis, "RED = MediaPipe detected (from AI render)", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, f"Adherence: {result.adherence_score:.0f}%  |  Error: {result.mean_error_px:.0f}px",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    vis_path = os.path.join(out_dir, "physics_adherence_overlay.png")
    cv2.imwrite(vis_path, vis)
    log.info("Overlay saved → %s", vis_path)

    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(vis_path)])


def _draw_mediapipe_skeleton(vis: np.ndarray, photo_rgb: np.ndarray,
                              w: int, h: int) -> None:
    """Detect and draw MediaPipe pose on the overlay image."""
    import mediapipe as mp

    model_path = os.path.join("tmp", "pose_landmarker_lite.task")
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(photo_rgb),
    )
    mp_result = landmarker.detect(mp_image)

    if not mp_result.pose_landmarks:
        landmarker.close()
        return

    lms = mp_result.pose_landmarks[0]
    mp_connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32),
    ]
    for a, b in mp_connections:
        lm_a, lm_b = lms[a], lms[b]
        if getattr(lm_a, "visibility", 1.0) > 0.3 and getattr(lm_b, "visibility", 1.0) > 0.3:
            cv2.line(vis, (int(lm_a.x * w), int(lm_a.y * h)),
                     (int(lm_b.x * w), int(lm_b.y * h)), (0, 0, 255), 2, cv2.LINE_AA)
    for lm in lms:
        if getattr(lm, "visibility", 1.0) > 0.3:
            cv2.circle(vis, (int(lm.x * w), int(lm.y * h)), 3, (0, 0, 255), -1)
    landmarker.close()


def main() -> None:
    out_dir = "outputs/controlnet_test"

    inputs = _load_inputs(out_dir)
    if inputs is None:
        return
    photo_rgb, physics_uv, w, h = inputs

    from src.modules.m5_physics_engine.physics_verifier import PhysicsAdherenceVerifier

    verifier = PhysicsAdherenceVerifier(threshold_px=30.0, img_size=(w, h))
    if not verifier.setup():
        log.error("MediaPipe setup failed")
        return

    result = verifier.verify_frame(photo_rgb, physics_uv, frame_index=0)
    verifier.cleanup()

    _print_report(result, w, h)
    _draw_overlay(photo_rgb, physics_uv, result, w, h, out_dir)


if __name__ == "__main__":
    main()

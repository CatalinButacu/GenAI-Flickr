"""M4 (Motion Generation) stage — retrieval, SSM, physics refinement, validation."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.shared.constants import GRAVITY, N_BODY_JOINTS
from src.shared.mem_profile import profile_memory, tracemalloc_snapshot
from src.shared.vocabulary import ACTIONS
from src.modules.motion import MotionGenerator
from src.modules.motion.models import MotionClip
from src.modules.motion.ssm_generator import SSMMotionGenerator, SSMMotionConfig
from src.shared.data.physics_dataset import extract_physics_state
from src.modules.motion.constants import MOTION_FPS

log = logging.getLogger(__name__)


def _lazy_validator():
    from scripts.body_validator import BodyValidator, repair_frame
    return BodyValidator, repair_frame


@dataclass
class ValidationReport:
    """Summary of biomechanical validation for one motion clip."""
    total_frames: int = 0
    valid_frames: int = 0
    repair_passes: int = 0
    violation_counts: dict = field(default_factory=dict)

    @property
    def validity_rate(self) -> float:
        return self.valid_frames / max(self.total_frames, 1)

    def __str__(self) -> str:
        return (
            f"{self.valid_frames}/{self.total_frames} frames valid "
            f"({self.validity_rate*100:.1f}%), "
            f"{self.repair_passes} repair pass(es)"
        )


class MotionStage:
    """M4: Generate motion clips from parsed scene actions."""

    def __init__(
        self, *,
        use_physics_ssm: bool = True,
        validate_motion: bool = True,
        max_repairs: int = 3,
        duration: float = 5.0,
        device: str = "cuda",
    ) -> None:
        self.use_physics_ssm = use_physics_ssm
        self.validate_motion = validate_motion
        self.max_repairs = max_repairs
        self.duration = duration
        self.device = device
        self.motion_gen: MotionGenerator | None = None
        self.physics_ssm_refiner: SSMMotionGenerator | None = None

    def setup(self) -> None:
        log.info("[M4] Initialising MotionGenerator (retrieval+SSM+semantic)")
        self.motion_gen = MotionGenerator(
            use_retrieval=True, use_ssm=True, use_semantic=True,
        )
        if self.use_physics_ssm:
            self._init_physics_ssm()
        else:
            log.info("[M4] PhysicsSSM disabled in config")

    # ── public API ──────────────────────────────────────────────────

    @profile_memory
    def run(self, parsed) -> dict[str, Any]:
        """Generate, refine, and validate motion clips for each actor."""
        if self.motion_gen is None:
            log.warning("[M4] motion generation not initialised")
            return {}

        total_frames = int(self.duration * MOTION_FPS)
        # Motion is generated at MOTION_FPS=30; physics/render output is at
        # DEFAULT_FPS=24.  Reconciliation happens implicitly in the physics stage
        # via render_interval = physics_hz // fps (240 // 24 = 10 steps/frame).
        log.info("[M4] target: %d frames (%.1fs x %dfps)",
                 total_frames, self.duration, MOTION_FPS)

        action_clips = self._collect_action_clips(parsed, total_frames)
        if not action_clips:
            log.warning("[M4] no action clips generated from parsed actions")
            return {}

        clips = self._sequence_actor_clips(action_clips)

        t0 = time.time()
        self._apply_physics_ssm(clips)
        log.info("[M4] PhysicsSSM refinement pass: %.2fs", time.time() - t0)

        if self.validate_motion:
            t0 = time.time()
            reports = self._validate_and_repair(clips)
            log.info("[M4] validation+repair pass: %.2fs", time.time() - t0)
            for actor, report in reports.items():
                log.info("[M4]   %s: %s", actor, report)

        return clips

    # ── internal helpers ────────────────────────────────────────────

    def _init_physics_ssm(self) -> None:
        cfg = SSMMotionConfig(use_physics=True)
        log.info("[M4] Creating PhysicsSSM refiner: d_physics=%d", cfg.d_physics)
        refiner = SSMMotionGenerator(
            backend="ssm_physics", config=cfg, device=self.device,
        )
        if refiner.setup():
            log.info("[M4] PhysicsSSM refinement layer active")
            self.physics_ssm_refiner = refiner
        else:
            log.warning("[M4] PhysicsSSM setup failed — skipping refinement")

    def _collect_action_clips(self, parsed, total_frames: int) -> list[tuple]:
        clips: list[tuple] = []
        for action in parsed.actions:
            act_def = ACTIONS.get(action.action_type)
            if not (act_def and act_def.motion_clip):
                log.warning("[M4] action '%s' has no motion_clip — skipping",
                            action.action_type)
                continue
            n = max(total_frames // max(len(parsed.actions), 1), 20)
            query = (action.raw_text or parsed.prompt
                     or act_def.motion_clip.replace("_", " "))
            t0 = time.time()
            clip = self.motion_gen.generate(query, num_frames=n)
            log.info("[M4] generate %r for '%s' -> %d frames (src=%s) in %.2fs",
                     query, action.actor, clip.num_frames, clip.source,
                     time.time() - t0)
            clips.append((action.actor, clip))
        return clips

    def _sequence_actor_clips(self, action_clips: list[tuple]) -> dict[str, Any]:
        actor_seqs: dict[str, list] = {}
        for actor, clip in action_clips:
            actor_seqs.setdefault(actor, []).append(clip)

        clips: dict[str, Any] = {}
        for actor, seq in actor_seqs.items():
            if len(seq) == 1:
                clips[actor] = seq[0]
            else:
                clips[actor] = _blend_clips(seq)
                log.info("[M4] sequenced %d clips for '%s' -> %d frames",
                         len(seq), actor, clips[actor].num_frames)
        return clips

    def _apply_physics_ssm(self, clips: dict[str, Any]) -> None:
        if not clips or self.physics_ssm_refiner is None:
            return
        for actor, clip in clips.items():
            if clip is None or clip.frames is None:
                continue
            physics_state = _build_physics_state(clip)
            refined = self.physics_ssm_refiner.generate(
                clip.action, num_frames=len(clip.frames),
                physics_state=physics_state,
            )
            if refined.raw_joints is None and clip.raw_joints is not None:
                refined.raw_joints = clip.raw_joints
            clips[actor] = refined
            log.info("[M4] PhysicsSSM refined '%s' -> %d frames",
                     actor, len(refined.frames))

    def _validate_and_repair(self, clips) -> dict[str, ValidationReport]:
        BodyValidator, repair_fn = _lazy_validator()
        validator = BodyValidator()
        reports: dict[str, ValidationReport] = {}
        for actor, clip in clips.items():
            if clip is None or clip.raw_joints is None:
                continue
            reports[actor] = self._repair_clip(
                actor, clip, validator, repair_fn,
            )
        return reports

    def _repair_clip(self, actor, clip, validator, repair_fn):
        # Validate body-only joints (N_BODY_JOINTS=22: pelvis + 21 body joints).
        # Full SMPL-X has N_JOINTS=55; hands/jaw/eyes are excluded from the
        # body-mechanics validator which expects the pyBullet-compatible chain.
        n_joints = clip.raw_joints.shape[1] if clip.raw_joints.ndim == 3 else 0
        if n_joints != N_BODY_JOINTS:
            return ValidationReport(total_frames=len(clip.raw_joints),
                                    valid_frames=len(clip.raw_joints))

        frames = [clip.raw_joints[i] for i in range(len(clip.raw_joints))]
        report = ValidationReport(total_frames=len(frames))

        for repair_pass in range(self.max_repairs + 1):
            seq_report = validator.evaluate_sequence(frames, verbose=False)
            report.valid_frames = seq_report.total_frames - seq_report.invalid_frames
            report.violation_counts = dict(seq_report.violation_counts)

            if seq_report.invalid_frames == 0:
                break
            if repair_pass >= self.max_repairs:
                log.warning("[VAL] %s: %.1f%% valid after %d repairs",
                            actor, report.validity_rate * 100, self.max_repairs)
                break

            repaired = 0
            for i, fr in enumerate(frames):
                fr_report = validator.evaluate_frame(fr, frame_idx=i)
                if not fr_report.valid:
                    frames[i] = repair_fn(fr, fr_report.violations)
                    repaired += 1
            report.repair_passes = repair_pass + 1

        clip.raw_joints = np.stack(frames, axis=0)
        return report


# ── module-level helpers ────────────────────────────────────────────────

def _build_physics_state(clip) -> np.ndarray:
    """Build 64-dim physics state from a motion clip.

    Clips are always 168-dim SMPL-X params (AMASS/InterX/PAHOI).
    Derives physics features via forward-kinematics helpers.
    """
    if clip.frames is not None and clip.frames.shape[-1] == 168:
        return extract_physics_state(clip.frames, fps=clip.fps, normalize=False)

    state = np.zeros((clip.frames.shape[0], 64), dtype=np.float32)
    state[:, 0] = GRAVITY
    if clip.raw_joints is not None:
        pelvis_h = clip.raw_joints[:, 0, 1]
        state[:, 1] = pelvis_h
        if len(pelvis_h) > 1:
            state[:, 2] = np.gradient(pelvis_h, 1.0 / clip.fps)
    return state


def _blend_clips(clips: list, blend_frames: int = 10) -> MotionClip:
    """Concatenate clips with linear cross-fade at boundaries."""
    feature_parts: list[np.ndarray] = []
    joint_parts: list[np.ndarray] = []
    labels: list[str] = []

    for i, clip in enumerate(clips):
        labels.append(clip.action)
        f, j = clip.frames, clip.raw_joints

        if i > 0 and blend_frames > 0:
            _crossfade_append(feature_parts, joint_parts, f, j, blend_frames)
        else:
            feature_parts.append(f)
            if j is not None:
                joint_parts.append(j)

    return MotionClip(
        action=" then ".join(labels),
        frames=np.concatenate(feature_parts, axis=0),
        fps=clips[0].fps,
        source="sequenced",
        raw_joints=(np.concatenate(joint_parts, axis=0)
                    if joint_parts and all(jp is not None for jp in joint_parts)
                    else None),
    )


def _crossfade_append(feature_parts, joint_parts, f, j, blend_frames):
    """Blend the last `blend_frames` of the previous clip with the first
    frames of the new clip using linear interpolation, avoiding abrupt
    transitions between consecutive action clips."""
    n = min(blend_frames, len(feature_parts[-1]), len(f))
    if n <= 0:
        feature_parts.append(f)
        if j is not None:
            joint_parts.append(j)
        return

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)

    tail, head = feature_parts[-1][-n:], f[:n]
    blended = tail * (1 - alpha[:, None]) + head * alpha[:, None]
    feature_parts[-1] = feature_parts[-1][:-n]
    feature_parts.append(blended)
    feature_parts.append(f[n:])

    if j is not None and joint_parts and joint_parts[-1] is not None:
        jt, jh = joint_parts[-1][-n:], j[:n]
        jb = jt * (1 - alpha[:, None, None]) + jh * alpha[:, None, None]
        joint_parts[-1] = joint_parts[-1][:-n]
        joint_parts.append(jb)
        joint_parts.append(j[n:])
    elif j is not None:
        joint_parts.append(j)

#!/usr/bin/env python3
"""Demo: Cinematic SMPL Action Sequences

Generates a multi-shot cinematic video showcasing SMPL body model
actions — walking, kicking, punching, jumping, turning — with
smooth camera orbits, zooms, and pitch sweeps.

Each *shot* has its own action, body type, camera choreography, and
title card.  The shots are concatenated into a single MP4.

Outputs (``outputs/cinematic_smpl/``):
  - ``cinematic_actions.mp4``     — full multi-shot video
  - ``poster_grid.png``           — 3 × 2 poster with key frames
  - ``shots/shot_N_title.mp4``    — individual shot videos
  - ``frames/shot_N/frame_XXXX.png`` — raw frames

Run:
    python scripts/demo_cinematic_smpl.py
"""
from __future__ import annotations

import math
import os
import sys
import logging
from collections.abc import Callable
from typing import Any

import cv2
import imageio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.physics.body_model import BodyParams, body_params_from_prompt
from src.modules.physics.silhouette_renderer import (
    SilhouetteSkeletonRenderer,
    SilhouetteProjector,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
log = logging.getLogger(__name__)

OUT_DIR = "outputs/cinematic_smpl"
FRAMES_DIR = os.path.join(OUT_DIR, "frames")
SHOTS_DIR = os.path.join(OUT_DIR, "shots")

WIDTH, HEIGHT = 1280, 720
FPS = 24


# ═══════════════════════════════════════════════════════════════════════════
# Easing functions
# ═══════════════════════════════════════════════════════════════════════════

def ease_in_out(t: float) -> float:
    """Smooth hermite ease-in / ease-out."""
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(t: float) -> float:
    return t ** 3


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ═══════════════════════════════════════════════════════════════════════════
# Biomechanical skeleton system
#
# All joint motion is done via ROTATIONS from the parent joint.
# Bone lengths are always preserved.  Joint angles are clamped to
# real human range-of-motion limits.
# ═══════════════════════════════════════════════════════════════════════════

# ── Anatomical rest-pose (arms at sides, natural standing) ──────────────

def _base_skeleton() -> np.ndarray:
    """Anatomical rest-pose skeleton (21 joints, Y-up mm).

    Joint positions are derived from the SMPL neutral model's actual
    rest joints (J_regressor @ v_template), centred at the pelvis and
    converted to millimetres.  Our convention is left = −X, right = +X
    (opposite to SMPL body-space), so all X values are sign-flipped
    relative to the raw SMPL JSON numbers.

    Arms are in the "arms-at-sides" position: the T-pose arm (which
    points horizontally outward in SMPL) is rotated 90 ° downward in
    the frontal (XY) plane so that the elbow sits below the shoulder
    and the wrist hangs at approximately hip level.

    SMPL rest joints used (neutral model, pelvis centred, mm):
      pelvis(0,0,0)  L_hip(+69.5,−91.4)  L_knee(+103.8,−466.6)
      L_ankle(+90.2,−864.6)  L_foot(+116.6,−920.4,+64.3)
      spine1(0,+108.9)  spine2(0,+244.1)  neck(0,+510.9)  head(0,+575.9)
      L_shoulder(+174.2,+449.3)  T-pose L_elbow(+433.9,+436.5)
    After 90° frontal rotation of T-pose arm → arms-at-sides:
      L_elbow(+162,+189)  L_wrist(+171,−61)
    All values below use our −X=left convention (X signs negated).
    """
    j = np.zeros((21, 3), dtype=np.float64)
    # ── Spine & head ──────────────────────────────────────────────
    j[0]  = [   0,    0,    0]   # 0  pelvis (world origin)
    j[1]  = [   0,  109,    0]   # 1  spine1  (SMPL joint 3)
    j[2]  = [   0,  244,    0]   # 2  chest   (SMPL joint 6 / spine2)
    j[3]  = [   0,  511,    0]   # 3  neck    (SMPL joint 12)
    j[4]  = [   0,  576,    0]   # 4  head top(SMPL joint 15)
    # ── Arms at sides (−X = left in our convention) ───────────────
    # Positions from SMPL T-pose joints (J_regressor @ v_template)
    # rotated 90° downward in the XY frontal plane.  Z offsets are
    # preserved from the T-pose (≈ −43 mm shoulder, −71 mm elbow/wrist)
    # so the SMPL rotation solver receives the correct bone directions.
    j[5]  = [-174,  449,  -43]   # 5  L shoulder (SMPL joint 16, T-pose Z)
    j[6]  = [-162,  190,  -71]   # 6  L elbow    (arms-at-sides, T-pose Z)
    j[7]  = [-170,  -60,  -72]   # 7  L wrist    (arms-at-sides, T-pose Z)
    j[8]  = [ 173,  448,  -48]   # 8  R shoulder (SMPL joint 17, T-pose Z)
    j[9]  = [ 160,  195,  -69]   # 9  R elbow    (arms-at-sides, T-pose Z)
    j[10] = [ 168,  -61,  -75]   # 10 R wrist    (arms-at-sides, T-pose Z)
    # ── Legs (straight stance, SMPL proportions) ──────────────────
    j[11] = [ -70,  -91,    0]   # 11 L hip    (SMPL joint 1)
    j[12] = [-104, -467,    0]   # 12 L knee   (SMPL joint 4)
    j[13] = [ -90, -865,    0]   # 13 L ankle  (SMPL joint 7)
    j[14] = [ -90, -880,  -30]   # 14 L heel   (behind ankle)
    j[15] = [-117, -920,   64]   # 15 L toe    (SMPL joint 10)
    j[16] = [  70,  -91,    0]   # 16 R hip    (SMPL joint 2)
    j[17] = [ 104, -473,    0]   # 17 R knee   (SMPL joint 5)
    j[18] = [  90, -871,    0]   # 18 R ankle  (SMPL joint 8)
    j[19] = [  90, -886,  -30]   # 19 R heel
    j[20] = [ 115, -920,   68]   # 20 R toe    (SMPL joint 11)
    return j


# ── Pre-compute bone lengths from rest pose ──────────────────────────────

_REST = _base_skeleton()

# SMPL 21-joint kinematic chains: (parent_idx, child_idx) for every bone
_BONE_PAIRS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # spine → head
    (2, 5), (5, 6), (6, 7),                    # L arm
    (2, 8), (8, 9), (9, 10),                   # R arm
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),  # L leg
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),  # R leg
]

# True rest-pose bone lengths (mm)
_BONE_LENGTHS: dict[tuple[int, int], float] = {}
for _p, _c in _BONE_PAIRS:
    _BONE_LENGTHS[(_p, _c)] = float(np.linalg.norm(_REST[_c] - _REST[_p]))


# ── Biomechanical joint angle limits (radians) ──────────────────────────
#
# Sagittal = YZ plane (flex/extend).  Frontal = XY plane (ab/adduction).
# Positive = forward (flex) or outward (abduct).
#
# Sources: Kapandji — Physiology of the Joints, NASA STD-3000.

_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    # sagittal (YZ) limits: (min, max) in radians
    "spine_yz":     (-0.25,  0.30),   # flex / extend
    "neck_yz":      (-0.40,  0.50),   # nod
    "shoulder_yz":  (-0.70,  2.80),   # extend behind ↔ flex overhead
    "elbow_yz":     ( 0.00,  2.50),   # only flexion (can't hyperextend)
    "hip_yz":       (-0.50,  2.00),   # extend behind ↔ flex forward
    "knee_yz":      ( 0.00,  2.40),   # only flexion
    "ankle_yz":     (-0.40,  0.60),   # dorsi / plantar flexion
    # frontal (XY) limits
    "spine_xy":     (-0.20,  0.20),   # lateral bend
    "shoulder_xy":  (-0.30,  2.60),   # adduct ↔ abduct overhead
    "hip_xy":       (-0.30,  0.80),   # adduct ↔ abduct
}


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ── Core rotation primitives (operate on skeleton array in-place) ────────

def _rotate_chain_yz(
    pose: np.ndarray,
    pivot_idx: int,
    child_indices: list[int],
    angle: float,
) -> None:
    """Rotate *child_indices* about *pivot_idx* in YZ plane (sagittal).

    This is the primary motion plane for walking, kicking, arm swing.
    Positive angle = forward flexion (leg forward, arm forward).
    Bone lengths are preserved (rotation, not translation).
    """
    origin = pose[pivot_idx].copy()
    c, s = math.cos(angle), math.sin(angle)
    for idx in child_indices:
        dy = pose[idx, 1] - origin[1]
        dz = pose[idx, 2] - origin[2]
        pose[idx, 1] = origin[1] + c * dy - s * dz
        pose[idx, 2] = origin[2] + s * dy + c * dz


def _rotate_chain_xy(
    pose: np.ndarray,
    pivot_idx: int,
    child_indices: list[int],
    angle: float,
) -> None:
    """Rotate *child_indices* about *pivot_idx* in XY plane (frontal).

    Used for ab/adduction — raising arms sideways, lateral leg spread.
    Positive angle = abduction (outward).
    """
    origin = pose[pivot_idx].copy()
    c, s = math.cos(angle), math.sin(angle)
    for idx in child_indices:
        dx = pose[idx, 0] - origin[0]
        dy = pose[idx, 1] - origin[1]
        pose[idx, 0] = origin[0] + c * dx - s * dy
        pose[idx, 1] = origin[1] + s * dx + c * dy


def _rotate_chain_xz(
    pose: np.ndarray,
    pivot_idx: int,
    child_indices: list[int],
    angle: float,
) -> None:
    """Rotate *child_indices* about *pivot_idx* around Y axis (axial/yaw).

    Used for torso twist, whole-body rotation, turning.
    """
    origin = pose[pivot_idx].copy()
    c, s = math.cos(angle), math.sin(angle)
    for idx in child_indices:
        dx = pose[idx, 0] - origin[0]
        dz = pose[idx, 2] - origin[2]
        pose[idx, 0] = origin[0] + c * dx - s * dz
        pose[idx, 2] = origin[2] + s * dx + c * dz


# ── Compound joint movers (with angle limits) ───────────────────────────

# Kinematic chain index lists
_L_ARM_CHAIN = [6, 7]      # children of L shoulder (5)
_R_ARM_CHAIN = [9, 10]     # children of R shoulder (8)
_L_FOREARM   = [7]         # children of L elbow (6)
_R_FOREARM   = [10]        # children of R elbow (9)
_L_LEG_CHAIN = [12, 13, 14, 15]  # children of L hip (11)
_R_LEG_CHAIN = [17, 18, 19, 20]  # children of R hip (16)
_L_LOWER_LEG = [13, 14, 15]      # children of L knee (12)
_R_LOWER_LEG = [18, 19, 20]      # children of R knee (17)
_L_FOOT      = [14, 15]    # children of L ankle (13)
_R_FOOT      = [19, 20]    # children of R ankle (18)
_SPINE_ABOVE_1 = list(range(2, 11))   # chest .. wrists
_SPINE_ABOVE_2 = list(range(3, 11))   # neck .. wrists
_HEAD_CHAIN    = [4]                   # head


def flex_shoulder(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Flex/extend shoulder in sagittal plane (clamped).

    angle > 0 → arm swings forward (flexion)
    angle < 0 → arm swings backward (extension)

    Note: negated internally because arm hangs in −Y direction; raw
    CCW YZ rotation (positive) would swing the arm backward.
    """
    lo, hi = _JOINT_LIMITS["shoulder_yz"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        _rotate_chain_yz(pose, 5, _L_ARM_CHAIN, -angle)
    else:
        _rotate_chain_yz(pose, 8, _R_ARM_CHAIN, -angle)


def abduct_shoulder(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Ab/adduct shoulder in frontal plane (clamped).

    angle > 0 → arm raises sideways/overhead
    angle < 0 → arm crosses body
    """
    lo, hi = _JOINT_LIMITS["shoulder_xy"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        _rotate_chain_xy(pose, 5, _L_ARM_CHAIN, -angle)  # mirror for left
    else:
        _rotate_chain_xy(pose, 8, _R_ARM_CHAIN, angle)


def flex_elbow(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Flex elbow (clamped: 0 = straight, 2.5 = fully bent).

    The elbow can only flex, never hyperextend.
    Negated internally for same reason as flex_shoulder.
    """
    lo, hi = _JOINT_LIMITS["elbow_yz"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        _rotate_chain_yz(pose, 6, _L_FOREARM, -angle)
    else:
        _rotate_chain_yz(pose, 9, _R_FOREARM, -angle)


def flex_hip(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Flex/extend hip in sagittal plane (clamped).

    angle > 0 → leg swings forward (flexion)
    angle < 0 → leg swings backward (extension)
    Negated internally — leg hangs in −Y, same issue as shoulders.
    """
    lo, hi = _JOINT_LIMITS["hip_yz"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        _rotate_chain_yz(pose, 11, _L_LEG_CHAIN, -angle)
    else:
        _rotate_chain_yz(pose, 16, _R_LEG_CHAIN, -angle)


def flex_knee(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Flex knee (clamped: 0 = straight, 2.4 = fully bent).

    The knee only flexes backward; it cannot hyperextend.
    """
    lo, hi = _JOINT_LIMITS["knee_yz"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        # Positive raw YZ rotates lower leg backward (−Z) — correct for knee bend.
        # flex_hip is negated but flex_knee is NOT, because backward is correct here.
        _rotate_chain_yz(pose, 12, _L_LOWER_LEG, angle)
    else:
        _rotate_chain_yz(pose, 17, _R_LOWER_LEG, angle)


def flex_ankle(
    pose: np.ndarray, side: str, angle: float,
) -> None:
    """Dorsi/plantar-flex ankle (clamped)."""
    lo, hi = _JOINT_LIMITS["ankle_yz"]
    angle = _clamp(angle, lo, hi)
    if side == "left":
        _rotate_chain_yz(pose, 13, _L_FOOT, angle)
    else:
        _rotate_chain_yz(pose, 18, _R_FOOT, angle)


def flex_spine(
    pose: np.ndarray, angle: float, level: int = 1,
) -> None:
    """Flex/extend spine at level 1 (lower) or 2 (upper).

    angle > 0 → lean forward,  angle < 0 → lean backward.
    """
    lo, hi = _JOINT_LIMITS["spine_yz"]
    angle = _clamp(angle, lo, hi)
    if level == 1:
        _rotate_chain_yz(pose, 1, _SPINE_ABOVE_1, angle)
    else:
        _rotate_chain_yz(pose, 2, _SPINE_ABOVE_2, angle)


def bend_spine_lateral(
    pose: np.ndarray, angle: float,
) -> None:
    """Lateral spine bend in frontal plane (clamped)."""
    lo, hi = _JOINT_LIMITS["spine_xy"]
    angle = _clamp(angle, lo, hi)
    _rotate_chain_xy(pose, 1, _SPINE_ABOVE_1, angle)


def twist_torso(
    pose: np.ndarray, angle: float,
) -> None:
    """Axial torso rotation (yaw) — limited to ±0.25 rad (~15°)."""
    angle = _clamp(angle, -0.25, 0.25)
    _rotate_chain_xz(pose, 1, _SPINE_ABOVE_1, angle)


def nod_head(
    pose: np.ndarray, angle: float,
) -> None:
    """Nod head forward/backward (clamped)."""
    lo, hi = _JOINT_LIMITS["neck_yz"]
    angle = _clamp(angle, lo, hi)
    _rotate_chain_yz(pose, 3, _HEAD_CHAIN, angle)


# ── Bone length enforcement ─────────────────────────────────────────────

def _enforce_bone_lengths(pose: np.ndarray) -> None:
    """Post-process: enforce rest-pose bone lengths on all bones.

    Walks each kinematic chain and projects child joints to the
    correct distance from their parent while preserving direction.
    """
    # Apply in kinematic order (parent before child)
    ordered_pairs = [
        # Spine
        (0, 1), (1, 2), (2, 3), (3, 4),
        # L arm
        (2, 5), (5, 6), (6, 7),
        # R arm
        (2, 8), (8, 9), (9, 10),
        # L leg
        (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
        # R leg
        (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    ]
    for p_idx, c_idx in ordered_pairs:
        target_len = _BONE_LENGTHS[(p_idx, c_idx)]
        bone_vec = pose[c_idx] - pose[p_idx]
        current_len = np.linalg.norm(bone_vec)
        if current_len < 1e-6:
            continue
        # Project child to correct distance along same direction
        correction = (target_len / current_len) - 1.0
        if abs(correction) > 0.001:  # only fix if > 0.1% deviation
            delta = bone_vec * correction
            # Move child and all its descendants
            descendants = _get_descendants(c_idx)
            for d in descendants:
                pose[d] += delta


def _get_descendants(joint_idx: int) -> list[int]:
    """Return joint_idx plus all descendants in kinematic tree."""
    # Hardcoded descendants for efficiency
    _desc: dict[int, list[int]] = {
        0: list(range(21)),
        1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        2: [2, 3, 4, 5, 6, 7, 8, 9, 10],
        3: [3, 4],
        4: [4],
        5: [5, 6, 7],
        6: [6, 7],
        7: [7],
        8: [8, 9, 10],
        9: [9, 10],
        10: [10],
        11: [11, 12, 13, 14, 15],
        12: [12, 13, 14, 15],
        13: [13, 14, 15],
        14: [14, 15],
        15: [15],
        16: [16, 17, 18, 19, 20],
        17: [17, 18, 19, 20],
        18: [18, 19, 20],
        19: [19, 20],
        20: [20],
    }
    return _desc.get(joint_idx, [joint_idx])


# ═══════════════════════════════════════════════════════════════════════════
# Standing idle — prepended to every action so the human is clearly
# visible in a natural standing pose before the action begins.
# ═══════════════════════════════════════════════════════════════════════════

STAND_FRAMES = 18  # ~0.75 s at 24 fps


def _generate_stand(n: int = STAND_FRAMES) -> list[np.ndarray]:
    """Generate *n* frames of relaxed standing with subtle idle sway.

    Includes micro-motions that make the pose look alive:
      - Gentle breathing (chest rise/fall)
      - Slight lateral weight shift
      - Tiny head nod oscillation
    """
    frames: list[np.ndarray] = []
    for i in range(n):
        t = i / max(n - 1, 1)
        p = _base_skeleton()

        # Breathing: subtle chest expansion (~5mm)
        breath = math.sin(t * 2 * math.pi) * 0.012
        flex_spine(p, breath, level=2)

        # Weight shift: very gentle lateral sway
        bend_spine_lateral(p, math.sin(t * math.pi) * 0.015)

        # Head: tiny nod (alive look)
        nod_head(p, math.sin(t * 1.5 * math.pi) * 0.03)

        _enforce_bone_lengths(p)
        frames.append(p)
    return frames


# ═══════════════════════════════════════════════════════════════════════════
# Action generators — all use rotation-based joint control
# Each starts with a standing intro via _generate_stand().
# ═══════════════════════════════════════════════════════════════════════════

def generate_walk(n_frames: int = 72, stride: float = 300.0, direction: float = 1.0) -> list[np.ndarray]:
    """Natural walking cycle with biomechanical joint rotations.

    Each joint is driven by sinusoidal oscillation within its anatomical
    range-of-motion.  Arm and leg swing are counter-phased, matching
    real human gait biomechanics.

    Parameters
    ----------
    direction : float
        1.0 = walk forward (positive Z), -1.0 = walk backward (negative Z).
        Backward walking reverses root translation, adjusts spinal lean, and
        inverts hip-swing phase to approximate a posterior gait pattern.
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / n_frames
        phase = 2 * math.pi * t
        p = _base_skeleton()

        # ── Hip flexion/extension (legs swing) ──
        # Backward walking: invert hip swing phase (legs lead behind the body)
        hip_swing = math.sin(phase) * 0.40 * direction
        flex_hip(p, "left",  hip_swing)
        flex_hip(p, "right", -hip_swing)

        # ── Knee flexion (stance leg bends, swing leg lifts) ──
        l_knee = max(0, -math.sin(phase)) * 0.45 + 0.08   # stance bend
        r_knee = max(0,  math.sin(phase)) * 0.45 + 0.08
        l_knee += max(0, math.sin(phase)) * 0.25           # swing lift
        r_knee += max(0, -math.sin(phase)) * 0.25
        flex_knee(p, "left",  l_knee)
        flex_knee(p, "right", r_knee)

        # ── Ankle (dorsiflexion in swing, plantarflexion at push-off) ──
        l_ankle = math.sin(phase + 0.3) * 0.15
        r_ankle = -math.sin(phase + 0.3) * 0.15
        flex_ankle(p, "left",  l_ankle)
        flex_ankle(p, "right", r_ankle)

        # ── Shoulder flex/extend (counter-swing to legs) ──
        arm_swing = math.sin(phase) * 0.35
        flex_shoulder(p, "left",  -arm_swing)
        flex_shoulder(p, "right",  arm_swing)

        # ── Elbow flexion (slight bend during swing) ──
        l_elbow = 0.25 + max(0, -math.sin(phase)) * 0.3
        r_elbow = 0.25 + max(0,  math.sin(phase)) * 0.3
        flex_elbow(p, "left",  l_elbow)
        flex_elbow(p, "right", r_elbow)

        # ── Torso counter-rotation ──
        twist_torso(p, math.sin(phase) * 0.06)

        # ── Spine flex: lean forward when walking forward, backward when walking back ──
        flex_spine(p, direction * 0.05, level=1)

        # ── Pelvis: vertical bob + lateral sway ──
        p[:, 1] += 12 * math.sin(2 * phase)      # bob at 2× gait freq
        bend_spine_lateral(p, math.sin(phase) * 0.04)  # hip drop

        # ── Root translation in travel direction ──
        z_off = direction * stride * t * 3
        p[:, 2] += z_off

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


def generate_kick(n_frames: int = 48) -> list[np.ndarray]:
    """Right-leg high kick with biomechanical constraints.

    Phases: guard pose → wind-up (crouch) → kick (hip flex + knee extend)
    → return.  All motion via joint rotations.
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        p = _base_skeleton()

        # ── Fighting guard: flex elbows, slight shoulder flex ──
        flex_shoulder(p, "left",  0.9)      # arms forward
        flex_shoulder(p, "right", 0.9)
        flex_elbow(p, "left",  1.8)         # bent up to guard
        flex_elbow(p, "right", 1.8)

        if t < 0.20:
            # Wind-up: crouch, weight on left leg
            s = ease_in_out(t / 0.20)
            flex_knee(p, "left",  0.25 * s)     # crouch (bend both knees)
            flex_knee(p, "right", 0.30 * s)
            flex_hip(p, "right", -0.20 * s)     # slight back-load
            flex_spine(p, 0.08 * s, level=1)   # lean forward slightly

        elif t < 0.55:
            # Kick: right hip flexes forward, knee extends (straightens)
            s = ease_out_cubic((t - 0.20) / 0.35)
            kick_hip = s * 1.8                   # ~103° hip flexion
            kick_knee = (1.0 - s) * 1.0          # knee straightens as kick extends
            flex_hip(p, "right", kick_hip)
            flex_knee(p, "right", kick_knee)
            # Standing leg: slight bend
            flex_knee(p, "left", 0.15)
            # Counter-balance: lean torso back
            flex_spine(p, -0.20 * s, level=1)
            # Open left arm for balance
            abduct_shoulder(p, "left", 0.6 * s)
            flex_elbow(p, "left",  1.2)  # loosen guard

        else:
            # Return: bring leg back
            s = ease_in_out((t - 0.55) / 0.45)
            remain = 1.0 - s
            flex_hip(p, "right", remain * 1.8)
            flex_knee(p, "right", s * 0.4)      # re-bend on landing
            flex_knee(p, "left", 0.15 * remain)
            flex_spine(p, -0.20 * remain, level=1)
            abduct_shoulder(p, "left", 0.6 * remain)
            flex_elbow(p, "left", 1.2 + 0.6 * s)

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


def generate_punch_combo(n_frames: int = 60) -> list[np.ndarray]:
    """Left jab → right cross → left hook.

    Punches are driven by shoulder flexion + elbow extension +
    torso rotation.  Bone lengths never change.
    """
    punch_events = [
        # (t_start, t_peak, t_end, type)
        (0.00, 0.12, 0.22, "left_jab"),
        (0.28, 0.40, 0.52, "right_cross"),
        (0.58, 0.72, 0.88, "left_hook"),
    ]
    frames: list[np.ndarray] = []

    for i in range(n_frames):
        t = i / (n_frames - 1)
        p = _base_skeleton()

        # ── Base fighting stance ──
        # Stagger feet: right foot forward
        p[16:21, 2] += 80    # R leg forward
        p[11:16, 2] -= 40    # L leg back
        flex_hip(p, "right", 0.08)
        flex_hip(p, "left", -0.06)
        flex_knee(p, "left",  0.15)
        flex_knee(p, "right", 0.12)

        # Guard position: shoulders flexed, elbows bent
        flex_shoulder(p, "left",  0.80)
        flex_shoulder(p, "right", 0.80)
        flex_elbow(p, "left",  1.90)
        flex_elbow(p, "right", 1.90)

        # ── Apply active punch ──
        for t_start, t_peak, t_end, punch_type in punch_events:
            if t < t_start or t > t_end:
                continue

            if t <= t_peak:
                s = ease_out_cubic((t - t_start) / max(t_peak - t_start, 0.01))
            else:
                s = 1.0 - ease_in_cubic((t - t_peak) / max(t_end - t_peak, 0.01))
            s = max(0.0, min(1.0, s))

            if punch_type == "left_jab":
                # Punch = shoulder flex forward + elbow extends
                # Override guard: flex shoulder more, straighten elbow
                flex_shoulder(p, "left", 0.80 + 1.2 * s)  # total ~2.0 rad
                flex_elbow(p, "left", 1.90 * (1.0 - 0.85 * s))  # nearly straight
                twist_torso(p, 0.10 * s)     # slight torso follow-through

            elif punch_type == "right_cross":
                # Power punch: big torso rotation + shoulder + straight arm
                flex_shoulder(p, "right", 0.80 + 1.4 * s)
                flex_elbow(p, "right", 1.90 * (1.0 - 0.90 * s))
                twist_torso(p, -0.20 * s)    # torso rotates into punch
                # Weight shifts forward
                flex_hip(p, "right", 0.08 + 0.10 * s)

            elif punch_type == "left_hook":
                # Hook: shoulder abducts + flexes at angle
                # Arm swings in an arc from the side
                abduct_shoulder(p, "left", 0.8 * s)
                flex_shoulder(p, "left", 0.80 + 0.8 * s)
                flex_elbow(p, "left", 1.90 * (1.0 - 0.3 * s))  # stays bent (hook)
                twist_torso(p, 0.20 * s)     # strong torso rotation

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


def generate_jump(n_frames: int = 48) -> list[np.ndarray]:
    """Vertical jump with biomechanical phases.

    Crouch (flex hips+knees+spine) → launch → air (tuck) → land (absorb).
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        p = _base_skeleton()

        if t < 0.22:
            # ── Crouch: deep knee+hip bend, arms back ──
            s = ease_in_out(t / 0.22)
            flex_hip(p, "left",  0.50 * s)
            flex_hip(p, "right", 0.50 * s)
            flex_knee(p, "left",  1.00 * s)
            flex_knee(p, "right", 1.00 * s)
            flex_spine(p, 0.20 * s, level=1)   # lean forward
            # Arms swing back for momentum
            flex_shoulder(p, "left",  -0.50 * s)
            flex_shoulder(p, "right", -0.50 * s)
            flex_elbow(p, "left",  0.4 * s)
            flex_elbow(p, "right", 0.4 * s)
            # Crouch lowers pelvis
            crouch_drop = s * 180  # mm
            p[:, 1] -= crouch_drop

        elif t < 0.55:
            # ── Airborne: parabolic height, legs tuck, arms up ──
            s = (t - 0.22) / 0.33
            height = 4 * s * (1 - s) * 350  # parabola, peak ~350mm
            p[:, 1] += height

            # Hip + knee tuck in air (bring legs up)
            tuck = math.sin(s * math.pi)
            flex_hip(p, "left",  0.35 * tuck)
            flex_hip(p, "right", 0.35 * tuck)
            flex_knee(p, "left",  0.60 * tuck)
            flex_knee(p, "right", 0.60 * tuck)

            # Arms sweep up (shoulder abduction + flexion)
            abduct_shoulder(p, "left",  1.2 * tuck)
            abduct_shoulder(p, "right", 1.2 * tuck)
            flex_shoulder(p, "left",  0.5 * tuck)
            flex_shoulder(p, "right", 0.5 * tuck)
            flex_elbow(p, "left",  0.3 * tuck)
            flex_elbow(p, "right", 0.3 * tuck)

        elif t < 0.75:
            # ── Landing: absorb impact with hip+knee bend ──
            s = ease_in_out((t - 0.55) / 0.20)
            flex_hip(p, "left",  0.40 * s)
            flex_hip(p, "right", 0.40 * s)
            flex_knee(p, "left",  0.80 * s)
            flex_knee(p, "right", 0.80 * s)
            flex_spine(p, 0.12 * s, level=1)
            flex_shoulder(p, "left",  0.3 * s)
            flex_shoulder(p, "right", 0.3 * s)
            flex_elbow(p, "left",  0.6 * s)
            flex_elbow(p, "right", 0.6 * s)
            # Landing drops pelvis
            p[:, 1] -= 120 * s

        else:
            # ── Recovery: stand back up ──
            s = ease_in_out((t - 0.75) / 0.25)
            remain = 1.0 - s
            flex_hip(p, "left",  0.40 * remain)
            flex_hip(p, "right", 0.40 * remain)
            flex_knee(p, "left",  0.80 * remain)
            flex_knee(p, "right", 0.80 * remain)
            flex_spine(p, 0.12 * remain, level=1)
            flex_shoulder(p, "left",  0.3 * remain)
            flex_shoulder(p, "right", 0.3 * remain)
            flex_elbow(p, "left",  0.6 * remain)
            flex_elbow(p, "right", 0.6 * remain)
            p[:, 1] -= 120 * remain

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


def generate_spin(n_frames: int = 48) -> list[np.ndarray]:
    """Full-body 360° spin (pirouette).

    Whole body rotates via root yaw.  Arms abduct outward during spin
    (centrifugal feel).  Slight jump at peak.
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        p = _base_skeleton()

        # ── Arms extend outward during spin ──
        ext = math.sin(t * math.pi)  # peaks at middle
        abduct_shoulder(p, "left",  1.8 * ext)
        abduct_shoulder(p, "right", 1.8 * ext)
        flex_elbow(p, "left",  0.3 * ext)
        flex_elbow(p, "right", 0.3 * ext)

        # ── Standing leg: slight bend ──
        flex_knee(p, "left",  0.10)
        flex_knee(p, "right", 0.10)

        # ── Slight rise at peak ──
        lift = math.sin(t * math.pi) * 40
        p[:, 1] += lift

        # ── Full-body yaw rotation (360°) ──
        angle = ease_in_out(t) * 2 * math.pi
        _rotate_chain_xz(p, 0, list(range(1, 21)), angle)

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


def generate_victory_pose(n_frames: int = 48) -> list[np.ndarray]:
    """Arms-up celebration — driven by shoulder abduction + flexion.

    Arms raise overhead via proper shoulder rotation, not translation.
    """
    frames: list[np.ndarray] = []
    for i in range(n_frames):
        t = i / (n_frames - 1)
        p = _base_skeleton()

        # Transition into pose over first 30%
        arm_t = min(1.0, t / 0.30)
        s = ease_in_out(arm_t)

        # ── Raise arms: abduct shoulders overhead ──
        abduct_shoulder(p, "left",  2.4 * s)     # arms go up
        abduct_shoulder(p, "right", 2.4 * s)
        # Slight shoulder flexion (arms angle forward a bit)
        flex_shoulder(p, "left",  0.3 * s)
        flex_shoulder(p, "right", 0.3 * s)
        # Elbows slightly bent (fist-pump look)
        flex_elbow(p, "left",  0.5 * s)
        flex_elbow(p, "right", 0.5 * s)

        # ── Chest out, slight lean back ──
        flex_spine(p, -0.10 * s, level=1)

        # ── Subtle breathing oscillation (after pose reached) ──
        if t > 0.30:
            breath = math.sin((t - 0.30) * 8 * math.pi) * 0.015
            flex_spine(p, breath, level=2)

        # ── Slight lateral weight shift ──
        bend_spine_lateral(p, math.sin(t * 2 * math.pi) * 0.03)

        _enforce_bone_lengths(p)
        frames.append(p)
    return _generate_stand() + frames


# ═══════════════════════════════════════════════════════════════════════════
# Camera choreography
# ═══════════════════════════════════════════════════════════════════════════

class CameraChoreography:
    """Define camera motion as (yaw, pitch, distance, target) over time."""

    def __init__(
        self,
        yaw_start: float, yaw_end: float,
        pitch_start: float, pitch_end: float,
        dist_start: float, dist_end: float,
        target_fn: Callable[..., Any] | None = None,
    ):
        self.yaw_s = yaw_start
        self.yaw_e = yaw_end
        self.pitch_s = pitch_start
        self.pitch_e = pitch_end
        self.dist_s = dist_start
        self.dist_e = dist_end
        self.target_fn = target_fn or (lambda t, sk: [0.0, 0.0, 0.0])

    def evaluate(self, t: float, skeleton: np.ndarray | None = None) -> dict:
        s = ease_in_out(t)
        return {
            "yaw": lerp(self.yaw_s, self.yaw_e, s),
            "pitch": lerp(self.pitch_s, self.pitch_e, s),
            "distance": lerp(self.dist_s, self.dist_e, s),
            "target": self.target_fn(t, skeleton),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Title card rendering
# ═══════════════════════════════════════════════════════════════════════════

def make_title_card(
    title: str,
    subtitle: str = "",
    duration_frames: int = 36,
    w: int = WIDTH,
    h: int = HEIGHT,
) -> list[np.ndarray]:
    """Create a cinematic title card with fade-in and fade-out."""
    frames: list[np.ndarray] = []
    for i in range(duration_frames):
        t = i / (duration_frames - 1) if duration_frames > 1 else 1.0
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Fade factor: fade in first 30%, hold, fade out last 30%
        if t < 0.30:
            alpha = ease_in_out(t / 0.30)
        elif t > 0.70:
            alpha = ease_in_out((1.0 - t) / 0.30)
        else:
            alpha = 1.0

        # Warm dark canvas
        canvas[:] = (int(18 * alpha), int(15 * alpha), int(12 * alpha))

        # Cinematic bars (letterbox)
        bar_h = int(h * 0.08)
        canvas[:bar_h] = 0
        canvas[h - bar_h:] = 0

        # Title text
        font_title = cv2.FONT_HERSHEY_SIMPLEX
        scale_t = max(1.2, w / 700.0)
        thick_t = max(2, int(scale_t * 2.5))
        (tw, th), _ = cv2.getTextSize(title.upper(), font_title, scale_t, thick_t)
        tx = (w - tw) // 2
        ty = h // 2 - 20
        color_t = tuple(int(c * alpha) for c in (235, 225, 210))
        cv2.putText(canvas, title.upper(), (tx, ty), font_title,
                    scale_t, color_t, thick_t, cv2.LINE_AA)

        # Subtitle
        if subtitle:
            scale_s = max(0.6, w / 1400.0)
            thick_s = max(1, int(scale_s * 2))
            (sw, sh), _ = cv2.getTextSize(subtitle, font_title, scale_s, thick_s)
            sx = (w - sw) // 2
            sy = ty + th + 30
            color_s = tuple(int(c * alpha) for c in (160, 150, 135))
            cv2.putText(canvas, subtitle, (sx, sy), font_title,
                        scale_s, color_s, thick_s, cv2.LINE_AA)

        # Decorative line under title
        line_w = int(tw * 0.6 * alpha)
        line_y = ty + 12
        lx = (w - line_w) // 2
        color_l = tuple(int(c * alpha) for c in (100, 85, 60))
        cv2.line(canvas, (lx, line_y), (lx + line_w, line_y), color_l, 2, cv2.LINE_AA)

        frames.append(canvas)
    return frames


# ═══════════════════════════════════════════════════════════════════════════
# Cinematic letterbox overlay
# ═══════════════════════════════════════════════════════════════════════════

def add_letterbox(frame: np.ndarray, bar_ratio: float = 0.07) -> np.ndarray:
    """Add cinematic letterbox bars (top + bottom)."""
    h, w = frame.shape[:2]
    bar_h = int(h * bar_ratio)
    out = frame.copy()
    out[:bar_h] = (out[:bar_h].astype(np.float32) * 0.15).astype(np.uint8)
    out[h - bar_h:] = (out[h - bar_h:].astype(np.float32) * 0.15).astype(np.uint8)
    return out


def add_shot_label(frame: np.ndarray, label: str) -> np.ndarray:
    """Overlay shot label on the frame (bottom-left)."""
    h, w = frame.shape[:2]
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, w / 2400.0)
    thick = max(1, int(scale * 2))
    text = label.upper()
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = 30
    y = h - int(h * 0.07) - 15
    # Subtle background
    cv2.rectangle(out, (x - 8, y - th - 6), (x + tw + 8, y + 6), (0, 0, 0), -1)
    cv2.putText(out, text, (x, y), font, scale, (180, 170, 155), thick, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Shot definitions
# ═══════════════════════════════════════════════════════════════════════════

def _track_pelvis_z(t: float, skeleton: np.ndarray | None) -> list[float]:
    """Camera target tracks forward-moving pelvis."""
    if skeleton is not None:
        return [0.0, 0.0, float(skeleton[0, 2]) / 1000.0]
    return [0.0, 0.0, 0.0]


SHOTS: list[dict] = [
    {
        "title": "The Walk",
        "subtitle": "Athletic stride — cinematic orbit",
        "prompt": "a tall athletic person walks forward",
        "action_label": "WALK FORWARD",
        "generator": lambda: generate_walk(72, stride=300.0),
        "camera": CameraChoreography(
            yaw_start=90, yaw_end=270,
            pitch_start=-15, pitch_end=-25,
            dist_start=4.5, dist_end=3.8,
            target_fn=_track_pelvis_z,
        ),
    },
    {
        "title": "The Kick",
        "subtitle": "High kick — dramatic close-up",
        "prompt": "a muscular martial artist kicks",
        "action_label": "HIGH KICK",
        "generator": lambda: generate_kick(48),
        "camera": CameraChoreography(
            yaw_start=70, yaw_end=110,
            pitch_start=-10, pitch_end=-30,
            dist_start=4.0, dist_end=3.0,
        ),
    },
    {
        "title": "The Combo",
        "subtitle": "Jab, cross, hook — boxing sequence",
        "prompt": "a heavyweight boxer throws punches",
        "action_label": "PUNCH COMBO",
        "generator": lambda: generate_punch_combo(60),
        "camera": CameraChoreography(
            yaw_start=30, yaw_end=60,
            pitch_start=-12, pitch_end=-20,
            dist_start=4.0, dist_end=3.2,
        ),
    },
    {
        "title": "The Jump",
        "subtitle": "Vertical leap — slow-motion arc",
        "prompt": "a young athletic person jumps",
        "action_label": "JUMP",
        "generator": lambda: generate_jump(48),
        "camera": CameraChoreography(
            yaw_start=45, yaw_end=135,
            pitch_start=-20, pitch_end=-35,
            dist_start=4.5, dist_end=3.5,
        ),
    },
    {
        "title": "The Spin",
        "subtitle": "Full-body rotation — tracking shot",
        "prompt": "a graceful dancer",
        "action_label": "SPIN",
        "generator": lambda: generate_spin(48),
        "camera": CameraChoreography(
            yaw_start=0, yaw_end=180,
            pitch_start=-18, pitch_end=-22,
            dist_start=4.0, dist_end=3.5,
        ),
    },
    {
        "title": "The Victory",
        "subtitle": "Celebration pose — hero shot",
        "prompt": "a tall muscular person celebrates",
        "action_label": "VICTORY",
        "generator": lambda: generate_victory_pose(48),
        "camera": CameraChoreography(
            yaw_start=20, yaw_end=-20,
            pitch_start=-10, pitch_end=-30,
            dist_start=5.0, dist_end=3.0,
        ),
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Render a single shot
# ═══════════════════════════════════════════════════════════════════════════

def render_shot(shot: dict, shot_idx: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Render one shot and return (frames_rgb, key_frame_bgr)."""
    title = shot["title"]
    prompt = shot["prompt"]
    action_label = shot["action_label"]
    camera: CameraChoreography = shot["camera"]

    log.info("  Generating action: %s", title)
    bp = body_params_from_prompt(prompt)
    skeletons = shot["generator"]()
    n = len(skeletons)
    log.info("    Body: height=%.2fm, muscle=%.1f, gender=%s → %d frames",
             bp.height_m, bp.muscle_mass, bp.gender, n)

    # Initial camera params from choreography at t=0
    cam0 = camera.evaluate(0.0, skeletons[0])
    renderer = SilhouetteSkeletonRenderer(
        img_w=WIDTH, img_h=HEIGHT,
        yaw_deg=cam0["yaw"],
        pitch_deg=cam0["pitch"],
        distance=cam0["distance"],
        target=cam0["target"],
        action_label=action_label,
        body_params=bp,
    )

    frames_rgb: list[np.ndarray] = []
    key_frame: np.ndarray | None = None
    frame_dir = os.path.join(FRAMES_DIR, f"shot_{shot_idx}")
    os.makedirs(frame_dir, exist_ok=True)

    for i, skeleton in enumerate(skeletons):
        t = i / max(n - 1, 1)
        cam_state = camera.evaluate(t, skeleton)

        renderer.update_camera(
            cam_state["yaw"],
            cam_state["pitch"],
            cam_state["distance"],
            cam_state["target"],
        )

        prev = skeletons[i - 1] if i > 0 else None
        frame = renderer.render_frame(skeleton, prev_xyz=prev)

        # Cinematic post-processing
        frame = add_letterbox(frame)
        frame = add_shot_label(frame, f"SHOT {shot_idx + 1}: {action_label}")

        # Save frame
        cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.png"), frame)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Capture key frame at peak action
        if key_frame is None and t >= 0.45:
            key_frame = frame.copy()

    if key_frame is None:
        key_frame = cv2.cvtColor(frames_rgb[n // 2], cv2.COLOR_RGB2BGR)

    return frames_rgb, key_frame


# ═══════════════════════════════════════════════════════════════════════════
# Create poster grid
# ═══════════════════════════════════════════════════════════════════════════

def create_poster(key_frames: list[np.ndarray]) -> str:
    """Arrange key frames in a 3×2 grid with labels."""
    # Resize each key frame to uniform size
    cell_w, cell_h = 640, 360
    cells: list[np.ndarray] = []
    for i, kf in enumerate(key_frames):
        resized = cv2.resize(kf, (cell_w, cell_h))
        # Add thin border
        cv2.rectangle(resized, (0, 0), (cell_w - 1, cell_h - 1), (40, 35, 30), 2)
        cells.append(resized)

    # Pad to 6 if needed
    while len(cells) < 6:
        cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row1 = np.concatenate(cells[:3], axis=1)
    row2 = np.concatenate(cells[3:6], axis=1)
    poster = np.concatenate([row1, row2], axis=0)

    # Title at top
    poster_with_header = np.zeros((poster.shape[0] + 80, poster.shape[1], 3), dtype=np.uint8)
    poster_with_header[:] = (18, 15, 12)
    poster_with_header[80:] = poster
    cv2.putText(poster_with_header, "CINEMATIC SMPL ACTIONS", (30, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (220, 210, 195), 3, cv2.LINE_AA)

    path = os.path.join(OUT_DIR, "poster_grid.png")
    cv2.imwrite(path, poster_with_header)
    log.info("Saved poster → %s", path)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SHOTS_DIR, exist_ok=True)

    log.info("=" * 70)
    log.info("  CINEMATIC SMPL ACTION DEMO")
    log.info("  %d shots — %dx%d @ %d fps", len(SHOTS), WIDTH, HEIGHT, FPS)
    log.info("=" * 70)

    all_frames: list[np.ndarray] = []
    key_frames: list[np.ndarray] = []

    # Opening title
    log.info("\n[Title] Opening card…")
    opening = make_title_card(
        "Cinematic SMPL Actions",
        "Pure NumPy body model — real-time rendering",
        duration_frames=48,
    )
    all_frames.extend([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in opening])

    for idx, shot in enumerate(SHOTS):
        log.info("\n[Shot %d/%d] %s — %s", idx + 1, len(SHOTS),
                 shot["title"], shot["subtitle"])

        # Shot title card
        title_frames = make_title_card(
            shot["title"], shot["subtitle"], duration_frames=24,
        )
        all_frames.extend([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in title_frames])

        # Render action
        shot_frames, key_frame = render_shot(shot, idx)
        all_frames.extend(shot_frames)
        key_frames.append(key_frame)

        # Save individual shot video
        shot_name = shot["title"].lower().replace(" ", "_")
        shot_path = os.path.join(SHOTS_DIR, f"shot_{idx}_{shot_name}.mp4")
        writer = imageio.get_writer(shot_path, fps=FPS, codec="libx264",
                                    quality=8, pixelformat="yuv420p")
        for sf in shot_frames:
            writer.append_data(sf)
        writer.close()
        log.info("    Saved shot video → %s", shot_path)

    # Closing title
    log.info("\n[Title] Closing card…")
    closing = make_title_card(
        "SMPL Body Model",
        "6890 vertices — 13776 faces — pure NumPy",
        duration_frames=48,
    )
    all_frames.extend([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in closing])

    # Write full video
    video_path = os.path.join(OUT_DIR, "cinematic_actions.mp4")
    log.info("\nWriting full video: %d frames → %s", len(all_frames), video_path)
    writer = imageio.get_writer(video_path, fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p")
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()

    # Poster
    poster_path = create_poster(key_frames)

    # Summary
    duration = len(all_frames) / FPS
    log.info("\n" + "=" * 70)
    log.info("  CINEMATIC DEMO COMPLETE")
    log.info("  ─ Full video:   %s  (%.1fs, %d frames)", video_path, duration, len(all_frames))
    log.info("  ─ Poster:       %s", poster_path)
    log.info("  ─ Individual:   %s/shot_*.mp4", SHOTS_DIR)
    log.info("  ─ Raw frames:   %s/shot_*/frame_*.png", FRAMES_DIR)
    log.info("  ─ %d shots × SMPL mesh → silhouette pipeline", len(SHOTS))
    log.info("=" * 70)
    print(f"\nOpen: {os.path.abspath(video_path)}")


if __name__ == "__main__":
    main()

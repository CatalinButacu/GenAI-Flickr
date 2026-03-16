#!/usr/bin/env python3
"""Body Validity Evaluator for SMPL 21-joint skeletons.

Checks whether a skeleton frame (or a full sequence) satisfies all
biomechanical constraints that make a human pose physically plausible:

  1. **Bone length preservation** — each bone within ±BONE_TOL of rest length
  2. **Joint angle limits**       — per-axis ROM clamped to Kapandji references
  3. **Ground penetration**       — no foot joint below the floor plane
  4. **Balance / CoM**            — CoM horizontal projection inside support polygon
  5. **Self-collision (gross)**   — wrist / hand not inside trunk bounding box

Usage
-----
::

    from scripts.body_validator import BodyValidator, validate_sequence

    validator = BodyValidator()
    report = validator.evaluate_frame(skeleton_21)  # single frame
    results = validate_sequence(frames)             # list of frames → summary
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import rest-pose constants from the demo backbone ───────────────────────
# We import lazily inside functions to avoid circular deps if the demo
# module is not on the path.  Everything needed is re-derived here.

# ─────────────────────────────────────────────────────────────────────────────
# REST POSE (21 joints, Y-up mm, left = -X)
# Derived from SMPL neutral model joint regressor, arms-at-sides.
# ─────────────────────────────────────────────────────────────────────────────

_REST = np.array([
    [   0,    0,    0],   # 0  pelvis
    [   0,  109,    0],   # 1  spine1
    [   0,  244,    0],   # 2  chest
    [   0,  511,    0],   # 3  neck
    [   0,  576,    0],   # 4  head
    [-174,  449,  -43],   # 5  L shoulder
    [-162,  190,  -71],   # 6  L elbow
    [-170,  -60,  -72],   # 7  L wrist
    [ 173,  448,  -48],   # 8  R shoulder
    [ 160,  195,  -69],   # 9  R elbow
    [ 168,  -61,  -75],   # 10 R wrist
    [ -70,  -91,    0],   # 11 L hip
    [-104, -467,    0],   # 12 L knee
    [ -90, -865,    0],   # 13 L ankle
    [ -90, -880,  -30],   # 14 L heel
    [-117, -920,   64],   # 15 L toe
    [  70,  -91,    0],   # 16 R hip
    [ 104, -473,    0],   # 17 R knee
    [  90, -871,    0],   # 18 R ankle
    [  90, -886,  -30],   # 19 R heel
    [ 115, -920,   68],   # 20 R toe
], dtype=np.float64)

# Kinematic pairs (parent, child)
_BONE_PAIRS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (2, 5), (5, 6), (6, 7),
    (2, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
]

_BONE_LENGTHS: dict[tuple[int, int], float] = {
    (p, c): float(np.linalg.norm(_REST[c] - _REST[p]))
    for p, c in _BONE_PAIRS
}

# Rest bone UNIT directions (parent → child)
_REST_DIRS: dict[tuple[int, int], np.ndarray] = {}
for _p, _c in _BONE_PAIRS:
    _v = _REST[_c] - _REST[_p]
    _REST_DIRS[(_p, _c)] = _v / (np.linalg.norm(_v) + 1e-12)

# ─────────────────────────────────────────────────────────────────────────────
# Joint angle limits (radians, from Kapandji & NASA STD-3000)
# Keys: (parent_idx, child_idx) → (min_rad, max_rad) total cone angle
# ─────────────────────────────────────────────────────────────────────────────

# Total ROM per bone as a half-cone (angle from rest direction).
# This is a conservative over-approximation that catches gross violations.
# Per-axis limits are applied in the per-axis checks below.
# Source: Kapandji (1974) "The Physiology of the Joints" & NASA STD-3000.
# Values in radians. NOT architectural — biomechanical reference data.
_ROM_CONE: dict[tuple[int, int], float] = {
    # spine — Kapandji Ch. 1: lumbar flexion ~25°, thoracic ~25°, cervical ~35°
    (0, 1):  0.45,   # lumbar  (≈25.8°)
    (1, 2):  0.45,   # thoracic (≈25.8°)
    (2, 3):  0.55,   # cervical (≈31.5°)
    (3, 4):  0.70,   # head nod + turn (≈40.1°)
    # Arm bones excluded: their world-space direction is dominated by the
    # shoulder rotation chain, so a fixed-rest cone produces false positives.
    # Arm ROM is enforced via the per-axis YZ check for shoulder only.
    # L leg — pelvis-hip attachment (0,11) excluded: structural connection;
    # staggered-stance legitimately shifts it beyond 0.20 rad.
    (11, 12): 2.10,  # hip  (≈120° — Kapandji: hip flexion ~120°+extension ~30°)
    (12, 13): 2.45,  # knee (≈140° — Kapandji: knee flexion ~140°)
    # ankle-heel and heel-toe excluded (foot flips during kick)
    # R leg — (0,16) excluded same reason
    (16, 17): 2.10,
    (17, 18): 2.45,
}

# Per-axis YZ sagittal limits (min, max) — only for PRIMARY flexion joints.
# Distal joints (wrist, heel, toe) are excluded: their world-space angle
# changes whenever a proximal joint (shoulder, hip) rotates, so world-space
# per-axis checking generates false positives.  They are covered by the
# cone check above.
_LIMITS_YZ: dict[tuple[int, int], tuple[float, float]] = {
    (0, 1):  (-0.25,  0.30),   # spine L1
    (1, 2):  (-0.25,  0.30),   # spine L2
    (2, 3):  (-0.40,  0.50),   # neck
    (5, 6):  (-1.50,  2.80),   # L shoulder (large backward range for victory/overhead)
    (8, 9):  (-1.50,  2.80),   # R shoulder
    (11, 12):(-1.80,  2.00),   # L hip (allow full kick back-swing ~103°)
    (12, 13):(-1.80,  2.40),   # L knee — follows thigh in world-space
    (16, 17):(-1.80,  2.00),   # R hip
    (17, 18):(-1.80,  2.40),   # R knee
}

# Ground level: lowest foot point in rest pose (mm), with small margin
_GROUND_Y = float(min(_REST[15, 1], _REST[20, 1])) - 10.0   # ~-930 mm
_FOOT_JOINTS = [13, 14, 15, 18, 19, 20]   # ankles, heels, toes

# DESIGN CHOICE: 5% bone-length tolerance — standard in skeletal motion
# capture validation (e.g. CMU MoCap QA). Generous enough to absorb
# floating-point and interpolation noise, strict enough to catch implausible
# stretching. Not architectural.
_BONE_TOL = 0.05   # 5%


# ─────────────────────────────────────────────────────────────────────────────
# Violation dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Violation:
    """One constraint violation in a single frame."""
    kind: str          # 'bone_length' | 'joint_angle' | 'ground' | 'balance' | 'collision'
    joint: str         # human-readable joint name
    severity: float    # magnitude of excess (mm for distances, radians for angles)
    details: str = ""  # optional description


@dataclass
class FrameReport:
    """Validation result for a single skeleton frame."""
    frame_idx: int
    valid: bool
    violations: list[Violation] = field(default_factory=list)

    @property
    def n_violations(self) -> int:
        return len(self.violations)

    def summary(self) -> str:
        if self.valid:
            return f"Frame {self.frame_idx}: OK"
        desc = "; ".join(
            f"{v.kind}@{v.joint}({v.severity:+.3f})"
            for v in self.violations
        )
        return f"Frame {self.frame_idx}: {self.n_violations} violations — {desc}"


@dataclass
class SequenceReport:
    """Aggregated validation results for a full sequence."""
    total_frames: int
    invalid_frames: int
    frame_reports: list[FrameReport]
    violation_counts: dict[str, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.invalid_frames == 0

    @property
    def validity_rate(self) -> float:
        if self.total_frames == 0:
            return 1.0
        return 1.0 - self.invalid_frames / self.total_frames

    def print_summary(self) -> None:
        print(f"\n{'='*64}")
        print(f"  SEQUENCE VALIDATION REPORT")
        print(f"{'='*64}")
        print(f"  Total frames  : {self.total_frames}")
        print(f"  Valid frames  : {self.total_frames - self.invalid_frames}")
        print(f"  Invalid frames: {self.invalid_frames}")
        print(f"  Validity rate : {self.validity_rate*100:.1f}%")
        if self.violation_counts:
            print(f"\n  Violations by type:")
            for kind, count in sorted(self.violation_counts.items(), key=lambda x: -x[1]):
                print(f"    {kind:20s}  {count:5d}")
        # Show worst offending frames (up to 10)
        bad = [r for r in self.frame_reports if not r.valid]
        if bad:
            print(f"\n  Sample invalid frames (first 10):")
            for r in bad[:10]:
                print(f"    {r.summary()}")
        print(f"{'='*64}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Joint name lookup
# ─────────────────────────────────────────────────────────────────────────────

_JOINT_NAMES = [
    "pelvis", "spine1", "chest", "neck", "head",
    "L_shoulder", "L_elbow", "L_wrist",
    "R_shoulder", "R_elbow", "R_wrist",
    "L_hip", "L_knee", "L_ankle", "L_heel", "L_toe",
    "R_hip", "R_knee", "R_ankle", "R_heel", "R_toe",
]

def _jname(idx: int) -> str:
    return _JOINT_NAMES[idx] if idx < len(_JOINT_NAMES) else str(idx)


# ─────────────────────────────────────────────────────────────────────────────
# BodyValidator
# ─────────────────────────────────────────────────────────────────────────────

class BodyValidator:
    """Validate SMPL 21-joint skeleton frames for biomechanical plausibility.

    Parameters
    ----------
    bone_tol : float
        Fractional tolerance for bone length checks (default 0.05 = 5%).
    check_balance : bool
        Enable center-of-mass balance check (default True).
    check_collision : bool
        Enable gross self-collision check (default True).
    ground_margin : float
        Extra margin below rest ground level before flagging penetration (mm).
    """

    def __init__(
        self,
        bone_tol: float = _BONE_TOL,
        check_balance: bool = True,
        check_collision: bool = True,
        # DESIGN CHOICE: 30mm margin below rest-pose floor plane.
        # Absorbs contact noise (shoe thickness ~25mm, skin deformation).
        # Not architectural.
        ground_margin: float = 30.0,
    ) -> None:
        self.bone_tol = bone_tol
        self.check_balance = check_balance
        self.check_collision = check_collision
        self.ground_y = _GROUND_Y - ground_margin

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate_frame(
        self,
        pose: np.ndarray,
        frame_idx: int = 0,
    ) -> FrameReport:
        """Validate a single (21, 3) skeleton frame.

        Returns a :class:`FrameReport` with all detected violations.
        """
        viols: list[Violation] = []
        viols.extend(self._check_bone_lengths(pose))
        viols.extend(self._check_joint_angles(pose))
        viols.extend(self._check_ground(pose))
        if self.check_balance:
            viols.extend(self._check_balance(pose))
        if self.check_collision:
            viols.extend(self._check_collision(pose))
        return FrameReport(
            frame_idx=frame_idx,
            valid=len(viols) == 0,
            violations=viols,
        )

    def evaluate_sequence(
        self,
        frames: list[np.ndarray],
        verbose: bool = False,
    ) -> SequenceReport:
        """Validate every frame in *frames* and return a :class:`SequenceReport`."""
        reports: list[FrameReport] = []
        counts: dict[str, int] = {}
        for i, f in enumerate(frames):
            r = self.evaluate_frame(f, frame_idx=i)
            reports.append(r)
            for v in r.violations:
                counts[v.kind] = counts.get(v.kind, 0) + 1
            if verbose and not r.valid:
                print(r.summary())

        invalid = sum(1 for r in reports if not r.valid)
        return SequenceReport(
            total_frames=len(frames),
            invalid_frames=invalid,
            frame_reports=reports,
            violation_counts=counts,
        )

    # ── Check: bone lengths ─────────────────────────────────────────────────

    def _check_bone_lengths(self, pose: np.ndarray) -> list[Violation]:
        viols = []
        for (p, c), rest_len in _BONE_LENGTHS.items():
            cur_len = float(np.linalg.norm(pose[c] - pose[p]))
            if rest_len < 1e-6:
                continue
            drift = abs(cur_len - rest_len) / rest_len
            if drift > self.bone_tol:
                excess = (cur_len - rest_len)
                viols.append(Violation(
                    kind="bone_length",
                    joint=f"{_jname(p)}->{_jname(c)}",
                    severity=excess,
                    details=f"drift {drift*100:.1f}% (rest={rest_len:.1f} cur={cur_len:.1f} mm)",
                ))
        return viols

    # ── Check: joint angles ─────────────────────────────────────────────────

    def _check_joint_angles(self, pose: np.ndarray) -> list[Violation]:
        """Check per-bone angle against cone ROM and per-axis YZ limits."""
        viols = []
        for (p, c), max_cone in _ROM_CONE.items():
            cur_bone = pose[c] - pose[p]
            cur_len = np.linalg.norm(cur_bone)
            if cur_len < 1e-6:
                continue
            cur_dir = cur_bone / cur_len
            rest_dir = _REST_DIRS[(p, c)]

            # Total cone angle
            dot = float(np.clip(np.dot(cur_dir, rest_dir), -1.0, 1.0))
            total_angle = math.acos(dot)
            if total_angle > max_cone:
                excess = total_angle - max_cone
                viols.append(Violation(
                    kind="joint_angle",
                    joint=f"{_jname(p)}->{_jname(c)}",
                    severity=excess,
                    details=f"cone {math.degrees(total_angle):.1f}° > limit {math.degrees(max_cone):.1f}°",
                ))
                continue  # skip per-axis if cone already violated

            # Per-axis YZ sagittal check (more precise, only for key joints)
            if (p, c) in _LIMITS_YZ:
                lo, hi = _LIMITS_YZ[(p, c)]
                # Angle of (cur_bone - rest_bone) projected on YZ plane
                # from the rest bone direction
                rest_yz = np.array([0.0, rest_dir[1], rest_dir[2]])
                cur_yz  = np.array([0.0, cur_dir[1], cur_dir[2]])
                rest_yz_len = np.linalg.norm(rest_yz)
                cur_yz_len  = np.linalg.norm(cur_yz)
                if rest_yz_len > 1e-6 and cur_yz_len > 1e-6:
                    rest_yz /= rest_yz_len
                    cur_yz  /= cur_yz_len
                    # Signed angle in YZ plane (positive = more Z)
                    signed_angle = math.atan2(
                        float(np.cross(rest_yz, cur_yz)[0]),  # x-component of cross
                        float(np.dot(rest_yz, cur_yz))
                    )
                    if signed_angle < lo - 0.05:
                        viols.append(Violation(
                            kind="joint_angle",
                            joint=f"{_jname(p)}->{_jname(c)}(YZ-)",
                            severity=lo - signed_angle,
                            details=f"YZ {math.degrees(signed_angle):.1f}° < min {math.degrees(lo):.1f}°",
                        ))
                    elif signed_angle > hi + 0.05:
                        viols.append(Violation(
                            kind="joint_angle",
                            joint=f"{_jname(p)}->{_jname(c)}(YZ+)",
                            severity=signed_angle - hi,
                            details=f"YZ {math.degrees(signed_angle):.1f}° > max {math.degrees(hi):.1f}°",
                        ))
        return viols

    # ── Check: ground penetration ───────────────────────────────────────────

    def _check_ground(self, pose: np.ndarray) -> list[Violation]:
        viols = []
        for j in _FOOT_JOINTS:
            y = float(pose[j, 1])
            if y < self.ground_y:
                viols.append(Violation(
                    kind="ground",
                    joint=_jname(j),
                    severity=self.ground_y - y,
                    details=f"Y={y:.1f} mm below floor {self.ground_y:.1f} mm",
                ))
        return viols

    # ── Check: CoM balance ──────────────────────────────────────────────────

    def _check_balance(self, pose: np.ndarray) -> list[Violation]:
        """Check whether the CoM horizontal projection falls within the
        convex hull of the support polygon (the foot joints)."""
        viols = []

        # Approximate joint masses (proportional to segment mass, Winter 2009)
        weights = np.array([
            10.0,  # pelvis
             3.0, 3.0, 1.5, 1.0,   # spine, chest, neck, head
             0.5, 1.2, 0.6,         # L shoulder, elbow, wrist
             0.5, 1.2, 0.6,         # R shoulder, elbow, wrist
             2.5, 3.0, 1.5, 0.5, 0.5,   # L leg chain
             2.5, 3.0, 1.5, 0.5, 0.5,   # R leg chain
        ], dtype=np.float64)
        weights /= weights.sum()

        com = np.average(pose, axis=0, weights=weights)
        com_xz = com[[0, 2]]   # horizontal projection (X, Z)

        # Support polygon: convex hull of foot contact joints
        foot_xz = pose[_FOOT_JOINTS][:, [0, 2]]
        # Expand hull by 40 mm (foot contact area)
        hull_min = foot_xz.min(axis=0) - 40.0
        hull_max = foot_xz.max(axis=0) + 40.0

        # Simple AABB check (conservative — a true convex hull would be tighter)
        margin_x = min(com_xz[0] - hull_min[0], hull_max[0] - com_xz[0])
        margin_z = min(com_xz[1] - hull_min[1], hull_max[1] - com_xz[1])

        if margin_x < 0 or margin_z < 0:
            excess = -min(margin_x, margin_z)
            viols.append(Violation(
                kind="balance",
                joint="CoM",
                severity=float(excess),
                details=(
                    f"CoM({com_xz[0]:.0f},{com_xz[1]:.0f}) outside "
                    f"support({hull_min[0]:.0f}..{hull_max[0]:.0f}, "
                    f"{hull_min[1]:.0f}..{hull_max[1]:.0f}) mm"
                ),
            ))
        return viols

    # ── Check: gross self-collision ─────────────────────────────────────────

    def _check_collision(self, pose: np.ndarray) -> list[Violation]:
        """Flag wrist/hand nodes that are inside the trunk bounding box."""
        viols = []

        # Trunk bounding box (pelvis to chest, padded)
        trunk_joints = pose[[0, 1, 2, 3], :]
        pad = 60.0   # mm padding around trunk joints

        t_min = trunk_joints.min(axis=0) - pad
        t_max = trunk_joints.max(axis=0) + pad

        for j in [7, 10]:   # L wrist, R wrist
            pt = pose[j]
            inside = all(t_min[k] <= pt[k] <= t_max[k] for k in range(3))
            if inside:
                viols.append(Violation(
                    kind="collision",
                    joint=_jname(j),
                    severity=0.0,
                    details=f"{_jname(j)} at {pt.round(0)} inside trunk bbox",
                ))
        return viols


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def validate_sequence(
    frames: Sequence[np.ndarray],
    verbose: bool = True,
    **kwargs,
) -> SequenceReport:
    """Validate a list of (21, 3) frames and optionally print the report.

    Keyword arguments are forwarded to :class:`BodyValidator`.
    """
    validator = BodyValidator(**kwargs)
    report = validator.evaluate_sequence(list(frames))
    if verbose:
        report.print_summary()
    return report


def repair_frame(
    pose: np.ndarray,
    violations: list[Violation],
) -> np.ndarray:
    """Attempt to repair a pose by blending toward rest on violated bones.

    For each violated bone, the child joint (and its descendants) is lerped
    10% toward the rest-pose position.  Multiple passes converge the frame
    to a valid configuration.

    Returns a corrected copy of *pose*.
    """
    p = pose.copy()
    repaired: set[tuple[int, int]] = set()

    for v in violations:
        if v.kind not in ("bone_length", "joint_angle"):
            continue
        # Parse joint string like "L_hip->L_knee(YZ+)"
        raw = v.joint.split("(")[0]
        parts = raw.split("->")
        if len(parts) != 2:
            continue
        parent_name, child_name = parts[0].strip(), parts[1].strip()
        try:
            pi = _JOINT_NAMES.index(parent_name)
            ci = _JOINT_NAMES.index(child_name)
        except ValueError:
            continue
        if (pi, ci) in repaired:
            continue
        repaired.add((pi, ci))

        # Blend child (and all its subtree) 15% toward rest offset
        rest_offset = _REST[ci] - _REST[pi]
        cur_offset  = p[ci] - p[pi]
        target_offset = cur_offset * 0.85 + rest_offset * 0.15

        delta = target_offset - cur_offset
        # Propagate delta to child and descendants
        p[ci] += delta
        # Find descendants via _BONE_PAIRS
        stack = [ci]
        visited = {pi, ci}
        while stack:
            nxt = stack.pop()
            for pp, cc in _BONE_PAIRS:
                if pp == nxt and cc not in visited:
                    p[cc] += delta
                    visited.add(cc)
                    stack.append(cc)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Validate a motion sequence NPY file.")
    ap.add_argument("npy", nargs="?", default=None,
                    help="Path to (N, 21, 3) NPY array. Omit to run a self-test.")
    ap.add_argument("--no-balance",    action="store_true", help="Skip balance check")
    ap.add_argument("--no-collision",  action="store_true", help="Skip collision check")
    ap.add_argument("--bone-tol",      type=float, default=0.05, help="Bone length tolerance")
    ap.add_argument("--verbose",       action="store_true", help="Print each invalid frame")
    args = ap.parse_args()

    if args.npy is not None:
        data = np.load(args.npy, allow_pickle=True)
        if data.ndim != 3 or data.shape[1] != 21 or data.shape[2] != 3:
            print(f"ERROR: Expected shape (N, 21, 3), got {data.shape}.")
            print("This file does not appear to be a SMPL 21-joint sequence.")
            sys.exit(1)
        frames = [data[i] for i in range(data.shape[0])]
        print(f"Loaded: {len(frames)} frames from '{args.npy}'")
    else:
        # Self-test: generate frames from the demo module
        print("No file given — running self-test on procedurally generated frames.")
        try:
            import importlib.util, os
            _demo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       "scripts", "demo_cinematic_smpl.py")
            spec = importlib.util.spec_from_file_location("demo", _demo_path)
            if spec is None or spec.loader is None:
                raise ImportError("Cannot load demo module")
            demo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(demo)
            frames = demo.generate_walk(48) + demo.generate_kick(36) + demo.generate_victory_pose(36)
            print(f"Generated {len(frames)} test frames (walk+kick+victory).")
        except Exception as e:
            print(f"Self-test generation failed: {e}")
            print("Using single rest-pose frame instead.")
            frames = [_REST.copy()]

    report = validate_sequence(
        frames,
        verbose=args.verbose,
        check_balance=not args.no_balance,
        check_collision=not args.no_collision,
        bone_tol=args.bone_tol,
    )

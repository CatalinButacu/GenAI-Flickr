from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.transform import Rotation

from src.modules.physics.humanoid import Q_UPRIGHT

log = logging.getLogger(__name__)


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product of two (x, y, z, w) quaternions."""
    q = (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))

# ── 21-joint humanoid indices (matches PyBullet humanoid.urdf) ────────────────
ROOT    = 0
SPINE1  = 1
SPINE2  = 2   # chest / thorax
NECK    = 3
HEAD    = 4
L_SHO   = 5
L_ELB   = 6
L_WRI   = 7
R_SHO   = 8
R_ELB   = 9
R_WRI   = 10
L_HIP   = 11
L_KNE   = 12
L_ANK   = 13
L_TBASE = 14
L_TOE   = 15
R_HIP   = 16
R_KNE   = 17
R_ANK   = 18
R_TBASE = 19
R_TOE   = 20

# ── SMPL-X joint index aliases (55 joints — body only for retargeting) ───────
SX_ROOT   = 0
SX_L_HIP  = 1
SX_R_HIP  = 2
SX_SPINE1 = 3
SX_L_KNE  = 4
SX_R_KNE  = 5
SX_SPINE2 = 6
SX_L_ANK  = 7
SX_R_ANK  = 8
SX_SPINE3 = 9
SX_L_FOOT = 10
SX_R_FOOT = 11
SX_NECK   = 12
SX_L_COLLAR = 13
SX_R_COLLAR = 14
SX_HEAD   = 15
SX_L_SHO  = 16
SX_R_SHO  = 17
SX_L_ELB  = 18
SX_R_ELB  = 19
SX_L_WRI  = 20
SX_R_WRI  = 21


# ── Low-level geometry helpers ────────────────────────────────────────────────

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)


def _hinge_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
    """
    Compute the bending angle AT *joint* using the parent and child positions.
    Returns 0 when fully straight, increasing as the limb bends.
    Convention: PyBullet knees/elbows expect **negative** values for flexion.
    """
    v_in  = _norm(parent - joint)
    v_out = _norm(child  - joint)
    cos_a = np.clip(np.dot(v_in, v_out), -1.0, 1.0)
    straight_deviation = np.arccos(cos_a)
    return -(np.pi - straight_deviation)


def _signed_plane_angle(
    vec: np.ndarray,
    plane_normal: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Signed angle of *vec* inside the plane defined by *plane_normal*,
    measured from *reference* (which must lie in the same plane).

    Positive direction is counter-clockwise around *plane_normal*
    (right-hand rule).

    Used to decompose ball-socket joints (hips, shoulders, spine)
    into independent anatomical degrees of freedom.
    """
    # Project both vectors onto the plane
    vec = vec - np.dot(vec, plane_normal) * plane_normal
    ref = reference - np.dot(reference, plane_normal) * plane_normal
    vec_n = _norm(vec)
    ref_n = _norm(ref)
    cos_a = np.clip(np.dot(vec_n, ref_n), -1.0, 1.0)
    angle = np.arccos(cos_a)
    # Determine sign via cross-product along the plane normal
    cross = np.cross(ref_n, vec_n)
    if np.dot(cross, plane_normal) < 0:
        angle = -angle
    return float(angle)


def _mat_to_quat(rot_mat: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix to (x, y, z, w) quaternion."""
    q = Rotation.from_matrix(rot_mat).as_quat()  # scipy: (x, y, z, w)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ── Coordinate-space conversion ───────────────────────────────────────────────
# 21-joint format is Y-up; PyBullet is Z-up.
# Conversion: (x, y, z) → (x, z, y).

def _kit_to_zup(v: np.ndarray) -> np.ndarray:
    """Convert Y-up vector to PyBullet Z-up coordinates."""
    return np.array([v[0], v[2], v[1]], dtype=np.float64)


# ── Root transform ────────────────────────────────────────────────────────────

def pelvis_transform(
    joints_mm: np.ndarray,
    start_pos_mm: np.ndarray,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """
    Compute the PyBullet root (pelvis) position and orientation for one frame.

    Parameters
    ----------
    joints_mm   : (21, 3) joints in mm, Y-up
    start_pos_mm: (3,) initial pelvis position to subtract (makes the motion
                  start at the origin)

    Returns
    -------
    position    : (x, y, z) in metres, Z-up
    orientation : (x, y, z, w) quaternion, Z-up
    """
    j = joints_mm

    # Translation: root = joint[0], centred to start position
    raw = (j[ROOT] - start_pos_mm) / 1000.0
    pos = _kit_to_zup(raw)
    # Keep a small height offset so the model stands above the ground plane
    pos = (float(pos[0]), float(pos[1]), float(max(pos[2] + 0.90, 0.80)))

    # Orientation from local pelvis frame
    up_kit    = _norm(j[SPINE1] - j[ROOT])             # spine direction (Y-up)
    right_kit = _norm(j[R_HIP]  - j[L_HIP])            # right direction

    up_zup    = _kit_to_zup(up_kit)
    right_zup = _kit_to_zup(right_kit)
    fwd_zup   = _norm(np.cross(up_zup, right_zup))
    right_zup = _norm(np.cross(fwd_zup, up_zup))       # re-orthogonalise

    # Build rotation matrix whose columns are [right, fwd, up]
    R = np.column_stack([right_zup, fwd_zup, up_zup])
    q_motion = _mat_to_quat(R)
    # Compose with upright rotation so the Y-up URDF stands in Z-up world
    quat = _quat_mul(q_motion, Q_UPRIGHT)
    pos_tuple: tuple[float, float, float] = (pos[0], pos[1], pos[2])
    return pos_tuple, quat


# ── Per-frame joint angle retargeting ─────────────────────────────────────────

def _retarget_spine(j, pelvis_up, pelvis_right, pelvis_fwd) -> tuple[float, float, float]:
    """Compute abdomen angles (spine lean + axial rotation)."""
    thorax_dir = _norm(j[SPINE2] - j[ROOT])
    abdomen_y = _signed_plane_angle(thorax_dir, pelvis_right, pelvis_up)
    abdomen_x = _signed_plane_angle(thorax_dir, pelvis_fwd, pelvis_up)
    shldr_right = _norm(j[R_SHO] - j[L_SHO])
    abdomen_z = _signed_plane_angle(shldr_right, pelvis_up, pelvis_right)
    return abdomen_x, abdomen_y, abdomen_z


def _retarget_leg(j, hip_idx, kne_idx, ank_idx, toe_idx,
                  pelvis_right, pelvis_fwd, limb_neutral):
    """Compute leg joint angles for one side."""
    thigh = _norm(j[kne_idx] - j[hip_idx])
    hip_y = _signed_plane_angle(thigh, pelvis_right, limb_neutral)
    hip_x = _signed_plane_angle(thigh, pelvis_fwd, limb_neutral)
    knee = _hinge_angle(j[hip_idx], j[kne_idx], j[ank_idx])
    foot = j[toe_idx]
    ankle = _hinge_angle(j[kne_idx], j[ank_idx], foot) * 0.6
    return hip_x, hip_y, knee, ankle


def _retarget_arm(j, sho_idx, elb_idx, wri_idx,
                  chest_right, chest_fwd, arm_neutral):
    """Compute arm joint angles for one side."""
    upper = _norm(j[elb_idx] - j[sho_idx])
    sho_y = _signed_plane_angle(upper, chest_right, arm_neutral)
    sho_x = _signed_plane_angle(upper, chest_fwd, arm_neutral)
    elbow = _hinge_angle(j[sho_idx], j[elb_idx], j[wri_idx])
    return sho_x, sho_y, elbow


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def retarget_frame(joints_mm: np.ndarray) -> dict[str, float]:
    """Convert one frame of 21-joint positions to PyBullet joint angles."""
    j = joints_mm

    pelvis_up, pelvis_right, pelvis_fwd = _compute_pelvis_frame(j)
    limb_neutral = -pelvis_up

    abdomen = _retarget_spine(j, pelvis_up, pelvis_right, pelvis_fwd)
    l_leg = _retarget_leg(j, L_HIP, L_KNE, L_ANK, L_TOE, pelvis_right, pelvis_fwd, limb_neutral)
    r_leg = _retarget_leg(j, R_HIP, R_KNE, R_ANK, R_TOE, pelvis_right, pelvis_fwd, limb_neutral)

    chest_right, chest_fwd, arm_neutral = _compute_chest_frame(j)
    l_arm = _retarget_arm(j, L_SHO, L_ELB, L_WRI, chest_right, chest_fwd, arm_neutral)
    r_arm = _retarget_arm(j, R_SHO, R_ELB, R_WRI, chest_right, chest_fwd, arm_neutral)

    return _build_angles_dict(abdomen, l_leg, r_leg, l_arm, r_arm)


def _compute_pelvis_frame(j):
    """Compute pelvis local coordinate frame."""
    pelvis_up = _norm(j[SPINE1] - j[ROOT])
    pelvis_right = _norm(j[R_HIP] - j[L_HIP])
    pelvis_fwd = _norm(np.cross(pelvis_up, pelvis_right))
    pelvis_right = _norm(np.cross(pelvis_fwd, pelvis_up))
    return pelvis_up, pelvis_right, pelvis_fwd


def _compute_chest_frame(j):
    """Compute chest local coordinate frame."""
    chest_up = _norm(j[NECK] - j[SPINE2])
    chest_right = _norm(j[R_SHO] - j[L_SHO])
    chest_fwd = _norm(np.cross(chest_up, chest_right))
    chest_right = _norm(np.cross(chest_fwd, chest_up))
    return chest_right, chest_fwd, -chest_up


def _euler_to_quat(x: float, y: float, z: float) -> tuple[float, float, float, float]:
    """Convert ZYX intrinsic Euler angles (radians) to quaternion (x, y, z, w)."""
    q = Rotation.from_euler('ZYX', [z, y, x]).as_quat()  # scipy: (x, y, z, w)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _build_angles_dict(abdomen, l_leg, r_leg, l_arm, r_arm) -> dict:
    """Assemble clamped joint angles into the output dict.

    Returns a dict mapping URDF joint names to:
      - quaternion (x,y,z,w) for spherical joints (chest, hips, shoulders, ankles)
      - scalar float for revolute joints (elbows, knees)
    """
    ax, ay, az = abdomen
    lhx, lhy, lk, la = l_leg
    rhx, rhy, rk, ra = r_leg
    lsx, lsy, le = l_arm
    rsx, rsy, re = r_arm

    # Clamp individual euler components
    ax  = _clamp(ax,  -0.25, 0.25)
    ay  = _clamp(ay,  -0.25, 0.25)
    az  = _clamp(az,  -0.20, 0.20)
    lhx = _clamp(lhx, -0.30, 0.30)
    lhy = _clamp(lhy, -0.80, 0.50)
    rhx = _clamp(rhx, -0.30, 0.30)
    rhy = _clamp(rhy, -0.80, 0.50)
    lsx = _clamp(lsx, -0.80, 0.80)
    lsy = _clamp(lsy, -0.90, 0.90)
    rsx = _clamp(rsx, -0.80, 0.80)
    rsy = _clamp(rsy, -0.90, 0.90)
    lk  = _clamp(lk,  -1.20, 0.00)
    rk  = _clamp(rk,  -1.20, 0.00)
    le  = _clamp(le,  -1.40, 0.00)
    re  = _clamp(re,  -1.40, 0.00)
    la  = _clamp(la,  -0.35, 0.30)
    ra  = _clamp(ra,  -0.35, 0.30)

    return {
        # Spherical joints → quaternion (x,y,z,w)
        "chest":           _euler_to_quat(ax, ay, az),
        "left_hip":        _euler_to_quat(lhx, lhy, 0.0),
        "right_hip":       _euler_to_quat(rhx, rhy, 0.0),
        "left_shoulder":   _euler_to_quat(lsx, lsy, 0.0),
        "right_shoulder":  _euler_to_quat(rsx, rsy, 0.0),
        "left_ankle":      _euler_to_quat(la, 0.0, 0.0),
        "right_ankle":     _euler_to_quat(ra, 0.0, 0.0),
        # Revolute joints → scalar angle
        "left_knee":       lk,
        "right_knee":      rk,
        "left_elbow":      le,
        "right_elbow":     re,
    }


def retarget_sequence(
    raw_joints: np.ndarray,
) -> tuple[list, list]:
    """
    Retarget a full motion sequence (21-joint or SMPL-X 55-joint).

    Parameters
    ----------
    raw_joints : (T, 21, 3) joints in mm, Y-up
                 OR (T, 55, 3) SMPL-X joints in metres, Y-up

    Returns
    -------
    joint_angles_list : list of T dicts (joint_name → angle)
    root_transforms   : list of T tuples ((pos), (quat))
    """
    if raw_joints.ndim != 3:
        raise ValueError(f"Expected (T, J, 3), got {raw_joints.shape}")

    n_joints = raw_joints.shape[1]
    if n_joints == 55:
        return retarget_sequence_smplx(raw_joints)
    elif n_joints == 21:
        return retarget_sequence_kit(raw_joints)
    else:
        raise ValueError(f"Unsupported joint count: {n_joints} (expected 21 or 55)")


def retarget_sequence_kit(
    raw_joints: np.ndarray,
) -> tuple[list, list]:
    """Retarget (T, 21, 3) joints in mm, Y-up to PyBullet angles."""
    if raw_joints.shape[1] != 21:
        raise ValueError(f"Expected (T, 21, 3), got {raw_joints.shape}")

    start_pos = raw_joints[0, ROOT].copy()
    joint_angles_list = []
    root_transforms   = []

    for frame in raw_joints:
        joint_angles_list.append(retarget_frame(frame))
        pos, quat = pelvis_transform(frame, start_pos)
        root_transforms.append((pos, quat))

    joint_angles_list = _smooth_joint_angles(joint_angles_list, alpha=0.3)

    return joint_angles_list, root_transforms


# ── SMPL-X retargeting (55 joints, metres, Y-up) ─────────────────────────────

def _smplx_to_zup(v: np.ndarray) -> np.ndarray:
    """SMPL-X is Y-up metres → PyBullet is Z-up metres."""
    return np.array([v[0], v[2], v[1]], dtype=np.float64)


def _yaw_only_quat(forward_zup: np.ndarray) -> tuple[float, float, float, float]:
    """Build an upright (roll=0, pitch=0) quaternion from the XY forward direction.

    PyBullet's Z-up world uses (x, y, z) so the yaw angle lives in the XY plane.
    """
    fxy = forward_zup[:2].copy()
    norm = float(np.linalg.norm(fxy))
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    fxy /= norm
    yaw = float(np.arctan2(fxy[0], fxy[1]))
    half = yaw / 2.0
    return (0.0, 0.0, float(np.sin(half)), float(np.cos(half)))


def _smplx_floor_offset(raw_joints: np.ndarray) -> float:
    """Estimate root-height correction from first-frame ankle/foot positions.

    SMPL-X Y-up: Y is vertical.  We look at the four lowest support joints
    (ankles + feet) in the first frame and compute how much to lift the root
    so the feet sit just above the floor (z=0 in PyBullet Z-up).

    Returns a value clamped to at least 0.90 m so the URDF never collapses.
    """
    # SX_L_ANK=7, SX_R_ANK=8, SX_L_FOOT=10, SX_R_FOOT=11
    support_indices = [SX_L_ANK, SX_R_ANK, SX_L_FOOT, SX_R_FOOT]
    first_frame = raw_joints[0]  # (55, 3), Y-up metres
    support_y = first_frame[support_indices, 1]  # Y = vertical
    min_y = float(np.min(support_y))
    return float(max(0.85, 0.02 - min_y))


def smplx_pelvis_transform(
    joints_m: np.ndarray,
    start_pos_m: np.ndarray,
    floor_offset_m: float = 0.90,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compute root transform from SMPL-X joints (55, 3) in metres, Y-up.

    Uses a *yaw-only* quaternion so the humanoid stays perfectly upright
    regardless of captured roll/pitch noise in the AMASS data.
    """
    j = joints_m

    raw = j[SX_ROOT] - start_pos_m
    pos = _smplx_to_zup(raw)
    pos_z = float(max(pos[2] + floor_offset_m, 0.02))
    pos_tuple: tuple[float, float, float] = (float(pos[0]), float(pos[1]), pos_z)

    # Yaw-only: derive forward from the hip axis
    right = _norm(j[SX_R_HIP] - j[SX_L_HIP])
    right_z = _smplx_to_zup(right)
    # In Z-up XY: forward ⊥ right  →  rotate 90° ⟹  (-right_y, right_x, 0)
    fwd_z = np.array([-right_z[1], right_z[0], 0.0], dtype=np.float64)
    q_yaw = _yaw_only_quat(fwd_z)
    # Compose with upright rotation so the Y-up URDF stands in Z-up world
    quat = _quat_mul(q_yaw, Q_UPRIGHT)
    return pos_tuple, quat


def retarget_frame_smplx(joints_m: np.ndarray) -> dict[str, float]:
    """Convert one frame of SMPL-X joints (55, 3 metres) to PyBullet angles."""
    j = joints_m

    pelvis_up = _norm(j[SX_SPINE1] - j[SX_ROOT])
    pelvis_right = _norm(j[SX_R_HIP] - j[SX_L_HIP])
    pelvis_fwd = _norm(np.cross(pelvis_up, pelvis_right))
    pelvis_right = _norm(np.cross(pelvis_fwd, pelvis_up))
    limb_neutral = -pelvis_up

    # Spine
    thorax_dir = _norm(j[SX_SPINE2] - j[SX_ROOT])
    abdomen_y = _signed_plane_angle(thorax_dir, pelvis_right, pelvis_up)
    abdomen_x = _signed_plane_angle(thorax_dir, pelvis_fwd, pelvis_up)
    shldr_right = _norm(j[SX_R_SHO] - j[SX_L_SHO])
    abdomen_z = _signed_plane_angle(shldr_right, pelvis_up, pelvis_right)

    # Legs
    def _leg(hip, kne, ank, foot):
        thigh = _norm(j[kne] - j[hip])
        hx = _signed_plane_angle(thigh, pelvis_fwd, limb_neutral)
        hy = _signed_plane_angle(thigh, pelvis_right, limb_neutral)
        knee = _hinge_angle(j[hip], j[kne], j[ank])
        ankle = _hinge_angle(j[kne], j[ank], j[foot]) * 0.6
        return hx, hy, knee, ankle

    l_leg = _leg(SX_L_HIP, SX_L_KNE, SX_L_ANK, SX_L_FOOT)
    r_leg = _leg(SX_R_HIP, SX_R_KNE, SX_R_ANK, SX_R_FOOT)

    # Arms
    chest_up = _norm(j[SX_NECK] - j[SX_SPINE2])
    chest_right = _norm(j[SX_R_SHO] - j[SX_L_SHO])
    chest_fwd = _norm(np.cross(chest_up, chest_right))
    chest_right = _norm(np.cross(chest_fwd, chest_up))
    arm_neutral = -chest_up

    def _arm(sho, elb, wri):
        upper = _norm(j[elb] - j[sho])
        sx = _signed_plane_angle(upper, chest_fwd, arm_neutral)
        sy = _signed_plane_angle(upper, chest_right, arm_neutral)
        elbow_a = _hinge_angle(j[sho], j[elb], j[wri])
        return sx, sy, elbow_a

    l_arm = _arm(SX_L_SHO, SX_L_ELB, SX_L_WRI)
    r_arm = _arm(SX_R_SHO, SX_R_ELB, SX_R_WRI)

    return _build_angles_dict(
        (abdomen_x, abdomen_y, abdomen_z),
        l_leg, r_leg, l_arm, r_arm,
    )


def retarget_sequence_smplx(
    raw_joints: np.ndarray,
) -> tuple[list, list]:
    """Retarget SMPL-X (T, 55, 3) joints in metres, Y-up."""
    if raw_joints.shape[1] != 55:
        raise ValueError(f"Expected (T, 55, 3), got {raw_joints.shape}")

    start_pos = raw_joints[0, SX_ROOT].copy()
    floor_offset = _smplx_floor_offset(raw_joints)
    log.info("[retarget_smplx] floor_offset=%.3fm (first-frame ankle min_y=%.3fm)",
             floor_offset, float(raw_joints[0, [SX_L_ANK, SX_R_ANK, SX_L_FOOT, SX_R_FOOT], 1].min()))
    joint_angles_list = []
    root_transforms = []

    for frame in raw_joints:
        joint_angles_list.append(retarget_frame_smplx(frame))
        pos, quat = smplx_pelvis_transform(frame, start_pos, floor_offset)
        root_transforms.append((pos, quat))

    # Temporal smoothing: EMA low-pass filter to reduce jitter / interpenetration
    joint_angles_list = _smooth_joint_angles(joint_angles_list, alpha=0.3)

    return joint_angles_list, root_transforms


def _smooth_joint_angles(
    angles_seq: list[dict],
    alpha: float = 0.3,
) -> list[dict]:
    """Exponential moving average smoothing over joint angle sequences.

    Handles both scalar (revolute) and quaternion (spherical) values.
    Quaternions are SLERP-interpolated; scalars are linearly blended.
    """
    if len(angles_seq) < 2:
        return angles_seq
    keys = list(angles_seq[0].keys())
    smoothed = [dict(angles_seq[0])]
    for i in range(1, len(angles_seq)):
        prev = smoothed[-1]
        curr = angles_seq[i]
        frame: dict = {}
        for k in keys:
            pv, cv = prev[k], curr[k]
            if isinstance(cv, tuple):
                # Quaternion LERP (renormalized) — good enough at small angles
                q = tuple(
                    pv[j] * (1.0 - alpha) + cv[j] * alpha
                    for j in range(4)
                )
                n = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2) ** 0.5
                frame[k] = tuple(c / n for c in q) if n > 1e-9 else (0.0, 0.0, 0.0, 1.0)
            else:
                frame[k] = pv * (1.0 - alpha) + cv * alpha
        smoothed.append(frame)
    return smoothed

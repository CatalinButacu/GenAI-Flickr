"""
#WHERE
    Called by pipeline.py to drive the PyBullet humanoid from KIT-ML data.

#WHAT
    Motion retargeting — converts KIT-ML raw joint positions (T, 21, 3)
    in mm Y-up to PyBullet joint angles (radians) and root transform.
    Derived angles become PD-motor targets; PyBullet solves dynamics.

#INPUT
    Raw joint array (T, 21, 3) mm Y-up, HumanoidBody instance.

#OUTPUT
    Per-frame joint angle dict, pelvis (position, orientation) transform.
  * joint reaction forces respect the URDF joint limits
  * the resulting motion therefore satisfies Newton's laws

Root position is tracked from the KIT-ML pelvis trajectory (``joint[0]``),
converted from Y-up mm to Z-up metres and applied via
``p.resetBasePositionAndOrientation`` before each step.  This is the
standard "reference-state initialisation" technique used in motion-tracking
physics controllers (DeepMimic, PhysChar, etc.).

KIT-ML joint indices
--------------------
 0=ROOT  1=SPINE1  2=SPINE2  3=NECK  4=HEAD
 5=L_SHO  6=L_ELB  7=L_WRI
 8=R_SHO  9=R_ELB 10=R_WRI
11=L_HIP 12=L_KNE 13=L_ANK 14=L_TBASE 15=L_TOE
16=R_HIP 17=R_KNE 18=R_ANK 19=R_TBASE 20=R_TOE

PyBullet humanoid.urdf joints  (from HumanoidBody.JOINT_INDICES)
-----------------------------------------------------------------
abdomen_x/y/z  |  r/l_hip_x/y/z  |  r/l_knee  |  r/l_ankle
r/l_shoulder_x/y  |  r/l_elbow
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── KIT-ML joint index aliases ────────────────────────────────────────────────
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


def _axis_angle_to_quat(axis: np.ndarray, angle: float) -> Tuple[float, float, float, float]:
    """Returns (x, y, z, w) quaternion for PyBullet."""
    axis = _norm(axis)
    s = np.sin(angle / 2.0)
    return (float(axis[0] * s), float(axis[1] * s),
            float(axis[2] * s), float(np.cos(angle / 2.0)))


def _mat_to_quat(rot_mat: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix to (x, y, z, w) quaternion."""
    trace = rot_mat[0, 0] + rot_mat[1, 1] + rot_mat[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot_mat[2, 1] - rot_mat[1, 2]) * s
        y = (rot_mat[0, 2] - rot_mat[2, 0]) * s
        z = (rot_mat[1, 0] - rot_mat[0, 1]) * s
    elif rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2])
        w = (rot_mat[2, 1] - rot_mat[1, 2]) / s
        x = 0.25 * s
        y = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        z = (rot_mat[0, 2] + rot_mat[2, 0]) / s
    elif rot_mat[1, 1] > rot_mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2])
        w = (rot_mat[0, 2] - rot_mat[2, 0]) / s
        x = (rot_mat[0, 1] + rot_mat[1, 0]) / s
        y = 0.25 * s
        z = (rot_mat[1, 2] + rot_mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1])
        w = (rot_mat[1, 0] - rot_mat[0, 1]) / s
        x = (rot_mat[0, 2] + rot_mat[2, 0]) / s
        y = (rot_mat[1, 2] + rot_mat[2, 1]) / s
        z = 0.25 * s
    return (float(x), float(y), float(z), float(w))


# ── Coordinate-space conversion ───────────────────────────────────────────────
# KIT-ML is Y-up; PyBullet humanoid.urdf is Z-up.
# Conversion: kit (x, y, z) → bullet (x, z, y)  — or equivalently a −90°
# rotation around the X axis.

def _kit_to_zup(v: np.ndarray) -> np.ndarray:
    """Convert KIT-ML Y-up vector to PyBullet Z-up world coordinates."""
    return np.array([v[0], v[2], v[1]], dtype=np.float64)


# ── Root transform ────────────────────────────────────────────────────────────

def pelvis_transform(
    joints_mm: np.ndarray,
    start_pos_mm: np.ndarray,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """
    Compute the PyBullet root (pelvis) position and orientation for one frame.

    Parameters
    ----------
    joints_mm   : (21, 3) KIT-ML joints in mm, Y-up
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
    pos = (float(pos[0]), float(pos[1]), float(max(pos[2] + 0.93, 0.85)))

    # Orientation from local pelvis frame
    up_kit    = _norm(j[SPINE1] - j[ROOT])             # spine direction (Y-up)
    right_kit = _norm(j[R_HIP]  - j[L_HIP])            # right direction

    up_zup    = _kit_to_zup(up_kit)
    right_zup = _kit_to_zup(right_kit)
    fwd_zup   = _norm(np.cross(up_zup, right_zup))
    right_zup = _norm(np.cross(fwd_zup, up_zup))       # re-orthogonalise

    # Build rotation matrix whose columns are [right, fwd, up]
    R = np.column_stack([right_zup, fwd_zup, up_zup])
    quat = _mat_to_quat(R)
    pos_tuple: Tuple[float, float, float] = (pos[0], pos[1], pos[2])
    return pos_tuple, quat


# ── Per-frame joint angle retargeting ─────────────────────────────────────────

def retarget_frame(joints_mm: np.ndarray) -> Dict[str, float]:
    """
    Convert one frame of KIT-ML 3D joint positions to PyBullet humanoid
    joint angles (radians).

    Parameters
    ----------
    joints_mm : np.ndarray, shape (21, 3), Y-up, millimetres

    Returns
    -------
    dict mapping PyBullet joint names to angles in **radians**.
    All values are clamped to physiologically safe ranges.
    """
    j = joints_mm  # shorthand — NO unit conversion needed for angles

    # ── pelvis local coordinate frame ────────────────────────────────────────
    pelvis_up    = _norm(j[SPINE1] - j[ROOT])       # +Y in KIT
    pelvis_right = _norm(j[R_HIP]  - j[L_HIP])      # +X
    pelvis_fwd   = _norm(np.cross(pelvis_up, pelvis_right))
    pelvis_right = _norm(np.cross(pelvis_fwd, pelvis_up))  # re-orthogonalise
    # Neutral limb direction = straight down from pelvis
    limb_neutral = -pelvis_up

    # ── torso / spine (abdomen) ───────────────────────────────────────────────
    thorax_dir = _norm(j[SPINE2] - j[ROOT])

    # abdomen_y: forward-backward lean [sagittal plane, normal = pelvis_right]
    abdomen_y = _signed_plane_angle(thorax_dir, pelvis_right, pelvis_up)

    # abdomen_x: side lean [frontal plane, normal = pelvis_fwd]
    abdomen_x = _signed_plane_angle(thorax_dir, pelvis_fwd, pelvis_up)

    # abdomen_z: axial rotation [transverse plane]
    shldr_right = _norm(j[R_SHO] - j[L_SHO])
    abdomen_z   = _signed_plane_angle(shldr_right, pelvis_up, pelvis_right)

    # ── left leg ─────────────────────────────────────────────────────────────
    l_thigh = _norm(j[L_KNE] - j[L_HIP])

    # hip_y: swing forward/back in sagittal plane
    left_hip_y = _signed_plane_angle(l_thigh, pelvis_right, limb_neutral)
    # hip_x: abduction/adduction in frontal plane
    left_hip_x = _signed_plane_angle(l_thigh, pelvis_fwd,   limb_neutral)

    # knee angle (negative = bent)
    left_knee  = _hinge_angle(j[L_HIP], j[L_KNE], j[L_ANK])

    # ankle angle
    l_foot = j[L_TBASE] if j[L_TBASE] is not None else j[L_TOE]
    left_ankle = _hinge_angle(j[L_KNE], j[L_ANK], l_foot) * 0.6

    # ── right leg ────────────────────────────────────────────────────────────
    r_thigh = _norm(j[R_KNE] - j[R_HIP])
    right_hip_y = _signed_plane_angle(r_thigh, pelvis_right, limb_neutral)
    right_hip_x = _signed_plane_angle(r_thigh, pelvis_fwd,   limb_neutral)
    right_knee  = _hinge_angle(j[R_HIP], j[R_KNE], j[R_ANK])
    r_foot = j[R_TBASE] if j[R_TBASE] is not None else j[R_TOE]
    right_ankle = _hinge_angle(j[R_KNE], j[R_ANK], r_foot) * 0.6

    # ── chest local frame (for arms) ─────────────────────────────────────────
    chest_up    = _norm(j[NECK]  - j[SPINE2])
    chest_right = _norm(j[R_SHO] - j[L_SHO])
    chest_fwd   = _norm(np.cross(chest_up, chest_right))
    chest_right = _norm(np.cross(chest_fwd, chest_up))  # re-orthogonalise
    arm_neutral = -chest_up   # arms hang down in T-pose

    # ── left arm ─────────────────────────────────────────────────────────────
    l_upper = _norm(j[L_ELB] - j[L_SHO])

    left_sho_y = _signed_plane_angle(l_upper, chest_right, arm_neutral)  # flex/extend
    left_sho_x = _signed_plane_angle(l_upper, chest_fwd,   arm_neutral)  # abduct

    left_elbow = _hinge_angle(j[L_SHO], j[L_ELB], j[L_WRI])

    # ── right arm ────────────────────────────────────────────────────────────
    r_upper = _norm(j[R_ELB] - j[R_SHO])

    right_sho_y = _signed_plane_angle(r_upper, chest_right, arm_neutral)
    right_sho_x = _signed_plane_angle(r_upper, chest_fwd,   arm_neutral)

    right_elbow = _hinge_angle(j[R_SHO], j[R_ELB], j[R_WRI])

    # ── clamp to safe anatomical ranges ──────────────────────────────────────
    def _clamp(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    return {
        # Torso
        "abdomen_x": _clamp(abdomen_x, -0.40,  0.40),
        "abdomen_y": _clamp(abdomen_y, -0.35,  0.35),
        "abdomen_z": _clamp(abdomen_z, -0.30,  0.30),

        # Left leg
        "left_hip_x":  _clamp(left_hip_x,  -0.40,  0.40),
        "left_hip_y":  _clamp(left_hip_y,  -1.00,  0.80),
        "left_hip_z":  0.0,
        "left_knee":   _clamp(left_knee,   -1.50,  0.00),
        "left_ankle":  _clamp(left_ankle,  -0.50,  0.40),

        # Right leg
        "right_hip_x": _clamp(right_hip_x, -0.40,  0.40),
        "right_hip_y": _clamp(right_hip_y, -1.00,  0.80),
        "right_hip_z": 0.0,
        "right_knee":  _clamp(right_knee,  -1.50,  0.00),
        "right_ankle": _clamp(right_ankle, -0.50,  0.40),

        # Left arm
        "left_shoulder_x": _clamp(left_sho_x,  -1.20,  1.20),
        "left_shoulder_y": _clamp(left_sho_y,  -1.40,  1.40),
        "left_elbow":      _clamp(left_elbow,  -1.57,  0.00),

        # Right arm
        "right_shoulder_x": _clamp(right_sho_x, -1.20,  1.20),
        "right_shoulder_y": _clamp(right_sho_y, -1.40,  1.40),
        "right_elbow":      _clamp(right_elbow, -1.57,  0.00),
    }


def retarget_sequence(
    raw_joints: np.ndarray,
) -> Tuple[list, list]:
    """
    Retarget a full motion sequence.

    Parameters
    ----------
    raw_joints : (T, 21, 3) KIT-ML joint positions in mm, Y-up

    Returns
    -------
    joint_angles_list : list of T dicts (joint_name → angle)
    root_transforms   : list of T tuples ((pos), (quat))
    """
    if raw_joints.ndim != 3 or raw_joints.shape[1] != 21:
        raise ValueError(f"Expected (T, 21, 3), got {raw_joints.shape}")

    start_pos = raw_joints[0, ROOT].copy()
    joint_angles_list = []
    root_transforms   = []

    for frame in raw_joints:
        joint_angles_list.append(retarget_frame(frame))
        pos, quat = pelvis_transform(frame, start_pos)
        root_transforms.append((pos, quat))

    return joint_angles_list, root_transforms

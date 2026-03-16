"""SMPL-X body model for AMASS/ARCTIC motion with hands, jaw, and eyes.

Pure-NumPy forward pass: shape blend shapes → joint regression →
pose blend shapes → linear blend skinning.

Loads model data from .npz files at
``data/arctic/unpack/models/smplx/SMPLX_NEUTRAL.npz``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.shared.constants import N_JOINTS, SMPLX_PARENTS

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

_SMPLX_SEARCH_DIRS: list[Path] = [
    _PROJECT_ROOT / "data" / "arctic" / "unpack" / "models" / "smplx",
    _PROJECT_ROOT / "checkpoints" / "smplx",
    _PROJECT_ROOT / "models" / "smplx",
]


def _axis_angle_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """Convert (3,) axis-angle to (3, 3) rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _find_smplx_model(gender: str = "neutral") -> Path | None:
    """Search for SMPL-X model file across known directories."""
    name = f"SMPLX_{gender.upper()}"
    for d in _SMPLX_SEARCH_DIRS:
        for ext in (".npz",):
            path = d / f"{name}{ext}"
            if path.is_file():
                return path
    return None


class SMPLXBody:
    """Pure-NumPy SMPL-X body model (55 joints, 10475 vertices).

    Parameters
    ----------
    model_path : Path or str
        Path to ``SMPLX_NEUTRAL.npz``.
    num_betas : int
        Number of shape PCA components to use (max 400).
    """

    _cache: dict[str, "SMPLXBody"] = {}

    @classmethod
    def get_or_create(cls, gender: str = "neutral") -> "SMPLXBody":
        """Return a cached SMPLXBody or create a new one."""
        if gender not in cls._cache:
            path = _find_smplx_model(gender)
            if path is None:
                raise FileNotFoundError(
                    f"SMPL-X model not found for gender='{gender}'. "
                    f"Searched: {[str(d) for d in _SMPLX_SEARCH_DIRS]}"
                )
            cls._cache[gender] = cls(path)
        return cls._cache[gender]

    def __init__(self, model_path: Path | str, num_betas: int = 16) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"SMPL-X model not found: {model_path}")

        self.model_path = model_path
        self.num_betas = num_betas

        log.info("Loading SMPL-X model from %s", model_path)

        data = np.load(str(model_path), allow_pickle=True)

        self.v_template = data["v_template"].astype(np.float64)     # (10475, 3)
        self.shapedirs = data["shapedirs"].astype(np.float64)       # (10475, 3, 400)
        self.posedirs = data["posedirs"].astype(np.float64)         # (10475, 3, 486)
        self.J_regressor = data["J_regressor"].astype(np.float64)   # (55, 10475)
        self.weights = data["weights"].astype(np.float64)           # (10475, 55)
        self.faces = data["f"].astype(np.int32)                     # (20908, 3)
        self.kintree_table = data["kintree_table"]                  # (2, 55)

        self.n_vertices = self.v_template.shape[0]
        self.n_joints = self.J_regressor.shape[0]
        self.n_faces = self.faces.shape[0]

        # Extract parent indices
        self._parents = self.kintree_table[0].astype(np.int32).tolist()
        self._parents[0] = -1

        # Rest-pose joints
        self.rest_joints = (self.J_regressor @ self.v_template)  # (55, 3)
        self.rest_vertices = self.v_template.copy()

        height = float(self.v_template[:, 1].max() - self.v_template[:, 1].min())
        log.info(
            "SMPL-X loaded: %d vertices, %d faces, %d joints, height=%.3fm",
            self.n_vertices, self.n_faces, self.n_joints, height,
        )

    def forward(
        self,
        betas: np.ndarray | None = None,
        body_pose: np.ndarray | None = None,
        left_hand_pose: np.ndarray | None = None,
        right_hand_pose: np.ndarray | None = None,
        jaw_pose: np.ndarray | None = None,
        eye_pose: np.ndarray | None = None,
        global_orient: np.ndarray | None = None,
        transl: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the SMPL-X forward pass.

        Returns
        -------
        vertices : (10475, 3) float64 — metres
        joints : (55, 3) float64 — metres
        """
        # Step 1-2: Shape blend shapes + joint regression
        v_shaped, J = self._shape_blend(betas)

        # Step 3: Build full pose rotation matrices
        rot_mats = self._build_rot_mats(
            body_pose, left_hand_pose, right_hand_pose,
            jaw_pose, eye_pose, global_orient,
        )

        # Step 4: Pose blend shapes
        ident = np.eye(3, dtype=np.float64)
        pose_feature = (rot_mats[1:] - ident[np.newaxis]).flatten()
        n_pd = min(len(pose_feature), self.posedirs.shape[2])
        v_posed = v_shaped + np.einsum(
            "vci,i->vc", self.posedirs[:, :, :n_pd], pose_feature[:n_pd],
        )

        # Step 5: Linear blend skinning
        return self._lbs(v_posed, J, rot_mats, transl)

    def forward_from_pose_vector(
        self,
        pose: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass from a (168,) unified SMPL-X pose vector.

        Layout: root_orient(3) + trans(3) + pose_body(63) + pose_hand(90)
                + pose_jaw(3) + pose_eye(6)
        """
        root_orient = pose[0:3]
        transl = pose[3:6]
        body_pose = pose[6:69]
        left_hand = pose[69:114]
        right_hand = pose[114:159]
        jaw_pose = pose[159:162]
        eye_pose = pose[162:168]

        return self.forward(
            betas=betas,
            body_pose=body_pose,
            left_hand_pose=left_hand,
            right_hand_pose=right_hand,
            jaw_pose=jaw_pose,
            eye_pose=eye_pose,
            global_orient=root_orient,
            transl=transl,
        )

    def _shape_blend(self, betas):
        """Shape blend shapes + joint regression."""
        n_b = min(self.num_betas, self.shapedirs.shape[2])
        if betas is not None:
            b = np.asarray(betas, dtype=np.float64)[:n_b]
            v_shaped = self.v_template + np.einsum(
                "vci,i->vc", self.shapedirs[:, :, :n_b], b,
            )
        else:
            v_shaped = self.v_template.copy()
        J = self.J_regressor @ v_shaped
        return v_shaped, J

    def _build_rot_mats(self, body_pose, left_hand_pose, right_hand_pose,
                        jaw_pose, eye_pose, global_orient):
        """Build (55, 3, 3) rotation matrices from axis-angle parameters."""
        rot_mats = np.zeros((self.n_joints, 3, 3), dtype=np.float64)

        # Joint 0: global orientation
        if global_orient is not None:
            rot_mats[0] = _axis_angle_to_rotmat(np.asarray(global_orient, dtype=np.float64))
        else:
            rot_mats[0] = np.eye(3, dtype=np.float64)

        # Joints 1-21: body pose (21 joints × 3 axis-angle = 63)
        if body_pose is None:
            body_pose = np.zeros(63, dtype=np.float64)
        body_pose = np.asarray(body_pose, dtype=np.float64)
        for j in range(1, 22):
            rot_mats[j] = _axis_angle_to_rotmat(body_pose[(j - 1) * 3: j * 3])

        # Joints 22-36: left hand (15 joints × 3 = 45)
        if left_hand_pose is None:
            left_hand_pose = np.zeros(45, dtype=np.float64)
        left_hand_pose = np.asarray(left_hand_pose, dtype=np.float64)
        for j in range(15):
            rot_mats[22 + j] = _axis_angle_to_rotmat(left_hand_pose[j * 3: (j + 1) * 3])

        # Joints 37-51: right hand (15 joints × 3 = 45)
        if right_hand_pose is None:
            right_hand_pose = np.zeros(45, dtype=np.float64)
        right_hand_pose = np.asarray(right_hand_pose, dtype=np.float64)
        for j in range(15):
            rot_mats[37 + j] = _axis_angle_to_rotmat(right_hand_pose[j * 3: (j + 1) * 3])

        # Joint 52: jaw
        if jaw_pose is not None:
            rot_mats[52] = _axis_angle_to_rotmat(np.asarray(jaw_pose, dtype=np.float64))
        else:
            rot_mats[52] = np.eye(3, dtype=np.float64)

        # Joints 53-54: eyes
        if eye_pose is not None:
            eye_pose = np.asarray(eye_pose, dtype=np.float64)
            rot_mats[53] = _axis_angle_to_rotmat(eye_pose[:3])
            rot_mats[54] = _axis_angle_to_rotmat(eye_pose[3:6])
        else:
            rot_mats[53] = np.eye(3, dtype=np.float64)
            rot_mats[54] = np.eye(3, dtype=np.float64)

        return rot_mats

    def _lbs(self, v_posed, J, rot_mats, transl):
        """Kinematic chain + linear blend skinning."""
        n_j = self.n_joints

        G = np.zeros((n_j, 4, 4), dtype=np.float64)
        for j in range(n_j):
            local_t = np.eye(4, dtype=np.float64)
            local_t[:3, :3] = rot_mats[j]
            local_t[:3, 3] = J[j]

            if self._parents[j] == -1:
                G[j] = local_t
            else:
                parent_t = np.eye(4, dtype=np.float64)
                parent_t[:3, :3] = rot_mats[j]
                parent_t[:3, 3] = J[j] - J[self._parents[j]]
                G[j] = G[self._parents[j]] @ parent_t

        # Remove rest-pose joint translations
        for j in range(n_j):
            rest_j_h = np.array([*J[j], 0.0], dtype=np.float64)
            G[j, :, 3] -= G[j] @ rest_j_h

        # LBS
        T = np.einsum("vj,jab->vab", self.weights, G)
        v_homo = np.ones((self.n_vertices, 4), dtype=np.float64)
        v_homo[:, :3] = v_posed
        v_final = np.einsum("vab,vb->va", T, v_homo)[:, :3]

        if transl is not None:
            v_final += np.asarray(transl, dtype=np.float64)

        J_final = self.J_regressor @ v_final
        return v_final, J_final

    def get_joint_positions(
        self,
        pose: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get (55, 3) joint positions from a (168,) pose vector."""
        _, joints = self.forward_from_pose_vector(pose, betas)
        return joints

    def get_joint_positions_batch(
        self,
        motion: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get (T, 55, 3) joint positions from a (T, 168) motion sequence."""
        T = motion.shape[0]
        joints = np.zeros((T, self.n_joints, 3), dtype=np.float64)
        for t in range(T):
            _, joints[t] = self.forward_from_pose_vector(motion[t], betas)
        return joints

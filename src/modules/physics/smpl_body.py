"""Pure-NumPy SMPL body model (Loper et al., 2015).

Forward pass: shape blendshapes → joint regression → pose blendshapes → LBS.
Falls back to analytical model if .npz files are unavailable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Model file search paths ─────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Search these directories (in order) for SMPL_NEUTRAL.npz / .pkl
_SEARCH_DIRS: list[Path] = [
    _PROJECT_ROOT / "checkpoints",
    _PROJECT_ROOT / "models" / "smpl",
    _PROJECT_ROOT / "models",
]

HAS_SMPL = True  # we only need numpy — always available

# 21-joint humanoid → SMPL 24-joint index mapping
KIT_TO_SMPL: dict[int, int] = {
    0: 0, 1: 3, 2: 6, 3: 12, 4: 15,
    5: 16, 6: 18, 7: 20,
    8: 17, 9: 19, 10: 21,
    11: 1, 12: 4, 13: 7, 15: 10,
    16: 2, 17: 5, 18: 8, 20: 11,
}

# SMPL kinematic tree: parents[j] = parent of joint j  (-1 = root)
SMPL_PARENTS: list[int] = [
    -1,  # 0  pelvis
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]


# ── Rotation helpers ─────────────────────────────────────────────────────────

def _rotation_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix that maps unit vector *v1* to *v2*.

    Uses Rodrigues' formula via the cross-product / dot-product of the
    two direction vectors.  Returns (3, 3) float64 rotation matrix.
    """
    v1 = v1 / max(np.linalg.norm(v1), 1e-12)
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)
    cross = np.cross(v1, v2)
    dot = float(np.dot(v1, v2))

    if dot > 1.0 - 1e-8:
        return np.eye(3)
    if dot < -1.0 + 1e-8:
        # 180° rotation — pick an arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(v1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(v1, perp)
        axis /= np.linalg.norm(axis)
        # R = 2 * outer(axis, axis) - I  (180° about axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew / (1.0 + dot)


def _rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Convert (3, 3) rotation matrix to (3,) axis-angle vector."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis /= max(np.linalg.norm(axis), 1e-12)
    return axis * angle


# ── SMPL body model (pure NumPy) ─────────────────────────────────────────────

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


def _find_model_file(gender: str = "neutral") -> Path | None:
    """Search for SMPL model file across known directories."""
    name = f"SMPL_{gender.upper()}"
    for d in _SEARCH_DIRS:
        for ext in (".npz", ".pkl"):
            path = d / f"{name}{ext}"
            if path.is_file():
                return path
    return None


class SMPLBody:
    """Pure-NumPy SMPL body model loaded from ``.npz`` files.

    Parameters
    ----------
    model_path : Path or str
        Path to ``SMPL_NEUTRAL.npz`` (or ``.pkl``).
    num_betas : int
        Number of shape PCA components to use (default 10, max 300).
    """

    _cache: dict[str, "SMPLBody"] = {}

    @classmethod
    def get_or_create(
        cls,
        gender: str = "neutral",
        model_path: Path | str | None = None,
    ) -> "SMPLBody":
        """Return a cached :class:`SMPLBody` or create a new one."""
        if model_path is None:
            found = _find_model_file(gender)
            if found is None:
                raise FileNotFoundError(
                    f"SMPL model file not found for gender='{gender}'.  "
                    f"Searched: {[str(d) for d in _SEARCH_DIRS]}.  "
                    "Download from https://smpl.is.tue.mpg.de/"
                )
            model_path = found
        key = str(model_path)
        if key not in cls._cache:
            cls._cache[key] = cls(model_path)
        return cls._cache[key]

    def __init__(
        self,
        model_path: Path | str,
        num_betas: int = 10,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"SMPL model not found: {model_path}")

        self.model_path = model_path
        self.num_betas = num_betas

        log.info("Loading SMPL model from %s", model_path)

        # Load model data
        data = np.load(str(model_path), allow_pickle=True)

        self.v_template: np.ndarray = data["v_template"].astype(np.float64)  # (6890, 3)
        self.shapedirs: np.ndarray = data["shapedirs"].astype(np.float64)    # (6890, 3, 300)
        self.posedirs: np.ndarray = data["posedirs"].astype(np.float64)      # (6890, 3, 207)
        self.J_regressor: np.ndarray = data["J_regressor"].astype(np.float64)  # (24, 6890)
        self.weights: np.ndarray = data["weights"].astype(np.float64)        # (6890, 24)
        self.faces: np.ndarray = data["f"].astype(np.int32)                  # (13776, 3)
        self.kintree_table: np.ndarray = data["kintree_table"]               # (2, 24)

        self.n_vertices = self.v_template.shape[0]
        self.n_faces = self.faces.shape[0]

        # Extract parent indices from kinematic tree
        self._parents = self.kintree_table[0].astype(np.int32).tolist()
        self._parents[0] = -1  # root has no parent

        # Cache rest-pose joints
        self.rest_joints: np.ndarray = (self.J_regressor @ self.v_template)  # (24, 3)
        self.rest_vertices: np.ndarray = self.v_template.copy()

        height = float(self.v_template[:, 1].max() - self.v_template[:, 1].min())
        log.info(
            "SMPL loaded: %d vertices, %d faces, template height=%.3fm",
            self.n_vertices, self.n_faces, height,
        )

    # ── SMPL forward pass ────────────────────────────────────────────────

    def forward(
        self,
        betas: np.ndarray | None = None,
        body_pose: np.ndarray | None = None,
        global_orient: np.ndarray | None = None,
        transl: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the SMPL forward pass (pure NumPy)."""
        v_shaped, J = self._shape_blend(betas)
        v_posed, rot_mats = self._pose_blend(v_shaped, body_pose, global_orient)
        return self._linear_blend_skinning(v_posed, J, rot_mats, transl)

    def _shape_blend(self, betas):
        """Step 1-2: Shape blend shapes + joint regression."""
        n_b = min(self.num_betas, self.shapedirs.shape[2])
        if betas is not None:
            b = np.asarray(betas, dtype=np.float64)[:n_b]
            v_shaped = self.v_template + np.einsum("vci,i->vc", self.shapedirs[:, :, :n_b], b)
        else:
            v_shaped = self.v_template.copy()
        J = self.J_regressor @ v_shaped
        return v_shaped, J

    def _pose_blend(self, v_shaped, body_pose, global_orient):
        """Step 3-4: Build rotation matrices + pose blend shapes."""
        if body_pose is None:
            body_pose = np.zeros(69, dtype=np.float64)
        if global_orient is None:
            global_orient = np.zeros(3, dtype=np.float64)

        rot_mats = np.zeros((24, 3, 3), dtype=np.float64)
        rot_mats[0] = _axis_angle_to_rotmat(global_orient)
        for j in range(1, 24):
            rot_mats[j] = _axis_angle_to_rotmat(body_pose[(j - 1) * 3: j * 3])

        ident = np.eye(3, dtype=np.float64)
        pose_feature = (rot_mats[1:] - ident[np.newaxis]).flatten()
        v_posed = v_shaped + np.einsum("vci,i->vc", self.posedirs, pose_feature)
        return v_posed, rot_mats

    def _linear_blend_skinning(self, v_posed, J, rot_mats, transl):
        """Step 5-7: Kinematic chain + LBS + translation."""
        G = np.zeros((24, 4, 4), dtype=np.float64)
        for j in range(24):
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

        for j in range(24):
            rest_j_h = np.array([*J[j], 0.0], dtype=np.float64)
            G[j, :, 3] -= G[j] @ rest_j_h

        T = np.einsum("vj,jab->vab", self.weights, G)
        v_homo = np.ones((self.n_vertices, 4), dtype=np.float64)
        v_homo[:, :3] = v_posed
        v_final = np.einsum("vab,vb->va", T, v_homo)[:, :3]

        if transl is not None:
            v_final += np.asarray(transl, dtype=np.float64)
        J_final = self.J_regressor @ v_final
        return v_final, J_final

    # ── Convenience methods ──────────────────────────────────────────────

    def get_shaped_mesh(
        self,
        betas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mesh vertices in T-pose with given shape.

        Returns
        -------
        vertices : (6890, 3) float  — metres
        faces : (F, 3) int32
        """
        verts, _ = self.forward(betas=betas)
        return verts, self.faces

    def get_posed_mesh(
        self,
        kit_joints_21: np.ndarray,
        betas: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return mesh vertices posed from 21-joint positions.

        Parameters
        ----------
        kit_joints_21 : (21, 3) float
            Joint positions in **Y-up millimetres** (21-joint format).
        betas : (10,) float or None
            Shape coefficients.

        Returns
        -------
        vertices : (6890, 3) float  — millimetres, Y-up
        faces : (F, 3) int32
        """
        # Map 21-joint → SMPL joint positions (metres)
        smpl_positions = self._kit_to_smpl_joints(kit_joints_21)

        # Get rest-pose joints for this shape
        if betas is not None:
            v_shaped = self.v_template + np.einsum(
                "vci,i->vc",
                self.shapedirs[:, :, :min(len(betas), self.shapedirs.shape[2])],
                np.asarray(betas, dtype=np.float64)[:self.shapedirs.shape[2]],
            )
            rest_joints = self.J_regressor @ v_shaped
        else:
            rest_joints = self.rest_joints

        # Compute pose from target joint positions
        body_pose, global_orient = self._compute_body_pose(smpl_positions, rest_joints)

        # Translation: match pelvis
        transl = smpl_positions[0] - rest_joints[0]

        # Forward pass
        verts, _ = self.forward(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
        )

        # metres → millimetres
        return verts * 1000.0, self.faces

    # ── Joint mapping ────────────────────────────────────────────────────

    def _kit_to_smpl_joints(self, kit_joints_21: np.ndarray) -> np.ndarray:
        """Map 21 joints → SMPL 24 joint positions (metres).

        Directly mapped joints use the 21-joint positions.  Missing SMPL
        joints (spine3, collars, hands) are interpolated.
        """
        kit = kit_joints_21.astype(np.float64) * 0.001  # mm → m
        smpl = np.zeros((24, 3), dtype=np.float64)

        # Direct mappings.
        # 21-joint format uses left = −X, right = +X.
        # SMPL uses left = +X, right = −X  (body-space convention).
        # Negate X on every copied joint so the rotation solver sees
        # bones pointing in the correct direction.
        _flip = np.array([-1.0, 1.0, 1.0], dtype=np.float64)
        for kit_idx, smpl_idx in KIT_TO_SMPL.items():
            smpl[smpl_idx] = kit[kit_idx] * _flip

        # Interpolated joints
        # 9  spine3:  midpoint(spine2, neck)
        smpl[9] = (smpl[6] + smpl[12]) * 0.5
        # 13 left_collar:  midway(spine3, left_shoulder)
        smpl[13] = smpl[9] * 0.4 + smpl[16] * 0.6
        # 14 right_collar: midway(spine3, right_shoulder)
        smpl[14] = smpl[9] * 0.4 + smpl[17] * 0.6
        # 22 left_hand:  extrapolate from elbow → wrist
        smpl[22] = smpl[20] + (smpl[20] - smpl[18]) * 0.3
        # 23 right_hand: extrapolate from elbow → wrist
        smpl[23] = smpl[21] + (smpl[21] - smpl[19]) * 0.3

        return smpl

    def _compute_body_pose(
        self,
        target_joints: np.ndarray,
        rest_joints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute SMPL pose (69,) + global orient (3,) from target joint positions."""
        # Primary outgoing bone for each joint (defines the rotation that
        # actually swings the limb).  Joints not listed here are leaf nodes
        # and stay at identity (no outgoing bone to orient).
        _PRIMARY: dict[int, int] = {
            # spine
            0: 3, 3: 6, 6: 9, 9: 12, 12: 15,
            # left leg
            1: 4, 4: 7, 7: 10,
            # right leg
            2: 5, 5: 8, 8: 11,
            # collars → shoulders
            13: 16, 14: 17,
            # left arm
            16: 18, 18: 20, 20: 22,
            # right arm
            17: 19, 19: 21, 21: 23,
        }

        # ── Pass 1: world rotation at each joint ────────────────────────
        # R_world[j] maps rest bone (j→child) to target bone (j→child).
        world_rots: list[np.ndarray] = [np.eye(3, dtype=np.float64)] * 24

        for j, c in _PRIMARY.items():
            rest_bone   = rest_joints[c]   - rest_joints[j]
            target_bone = target_joints[c] - target_joints[j]

            rest_len   = np.linalg.norm(rest_bone)
            target_len = np.linalg.norm(target_bone)
            if rest_len < 1e-8 or target_len < 1e-8:
                continue

            world_rots[j] = _rotation_between(
                rest_bone / rest_len,
                target_bone / target_len,
            )

        # ── Pass 2: localise into parent frame ──────────────────────────
        local_rots: list[np.ndarray] = [np.eye(3, dtype=np.float64)] * 24
        for j in range(1, 24):
            p = SMPL_PARENTS[j]
            local_rots[j] = world_rots[p].T @ world_rots[j]

        # ── Pass 3: encode as axis-angle ────────────────────────────────
        body_pose = np.zeros(69, dtype=np.float64)
        for j in range(1, 24):
            body_pose[(j - 1) * 3: j * 3] = _rotmat_to_axis_angle(local_rots[j])

        global_orient = _rotmat_to_axis_angle(world_rots[0])
        return body_pose, global_orient


# ── BodyParams → SMPL betas mapping ─────────────────────────────────────────

if TYPE_CHECKING:
    from src.modules.physics.body_model import BodyParams


def smpl_betas_from_body_params(params: "BodyParams") -> np.ndarray:
    """Convert BodyParams to SMPL's 10 PCA shape betas."""
    from .body_model import BodyParams  # avoid circular import

    betas = np.zeros(10, dtype=np.float32)
    _set_shape_betas(betas, params)
    _apply_demographic_betas(betas, params)
    return betas


def _set_shape_betas(betas: np.ndarray, params) -> None:
    """Fill β₀–β₇ from body shape parameters."""
    betas[0] = ((params.height_m - 1.72) / 0.10) * 1.5
    betas[1] = (params.bulk_factor - 1.0) * 3.0
    betas[2] = (params.shoulder_width - 1.0) * 2.5
    betas[3] = (params.body_fat - 1.0) * 0.4 - 0.3
    betas[4] = (params.hip_width - 1.0) * 2.5
    betas[5] = (params.leg_length - 1.0) * 1.5
    betas[6] = (params.arm_length - 1.0) * 1.0
    betas[7] = (params.muscle_mass - 1.0) * 1.5


def _apply_demographic_betas(betas: np.ndarray, params) -> None:
    """Apply gender and age adjustments to β₈–β₉ and fine-tune β₀/β₃."""
    match params.gender:
        case "male":
            betas[8] = 0.8
            betas[9] = -0.3
        case "female":
            betas[8] = -0.8
            betas[9] = 0.3
    match params.age_group:
        case "child":
            betas[0] -= 2.0
            betas[1] -= 0.5
        case "elderly":
            betas[0] -= 0.3
            betas[3] += 0.5


# ── Mesh → silhouette rendering ─────────────────────────────────────────────

def render_smpl_silhouette(
    vertices_3d: np.ndarray,
    faces: np.ndarray,
    project_fn: Callable[..., Any],
    canvas: np.ndarray,
    *,
    fill_color: tuple[int, int, int] = (180, 160, 140),
    outline_color: tuple[int, int, int] = (40, 35, 30),
    outline_thickness: int = 1,
    shadow: bool = True,
) -> np.ndarray:
    """Render an SMPL mesh as a filled silhouette on *canvas*.

    Projects all mesh vertices to 2D via *project_fn*, then rasterises
    the **front-facing** triangles as a filled mask.

    Parameters
    ----------
    vertices_3d : (V, 3) float
        Mesh vertex positions (Y-up mm or any consistent unit).
    faces : (F, 3) int
        Triangle face indices.
    project_fn : callable
        ``project_fn(xyz_N3) → uvd_N3``  — projects (N, 3) world coords
        to (N, 3) screen coords ``(col, row, depth)``.
    canvas : (H, W, 3) uint8
        Target image (modified in-place and returned).
    fill_color : (B, G, R)
        Silhouette fill colour.
    outline_color : (B, G, R)
        Thin outline for definition.
    outline_thickness : int
        Outline width (0 to disable).
    shadow : bool
        Draw ground shadow.

    Returns
    -------
    canvas : (H, W, 3) uint8
    """
    h, w = canvas.shape[:2]

    # 1. Project all vertices to 2D
    projected = project_fn(vertices_3d)  # (V, 3): col, row, depth
    xy = projected[:, :2]  # (V, 2)

    # 2. Get triangle screen coords
    tri_2d = xy[faces]  # (F, 3, 2)

    # 3. Back-face culling (vectorised cross-product)
    v1 = tri_2d[:, 1] - tri_2d[:, 0]  # (F, 2)
    v2 = tri_2d[:, 2] - tri_2d[:, 0]  # (F, 2)
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # (F,)
    front_mask = cross > 0  # front-facing triangles

    front_tris = tri_2d[front_mask].astype(np.int32)  # (F', 3, 2)

    # 4. Rasterise front-facing triangles into a mask
    mask = np.zeros((h, w), dtype=np.uint8)
    for tri in front_tris:
        cv2.fillConvexPoly(mask, tri, (255,))

    # 5. Ground shadow
    if shadow:
        _draw_smpl_shadow(canvas, xy, w, h)

    # 6. Smooth mask edges
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.8)

    # 7. Composite silhouette onto canvas
    alpha = mask.astype(np.float32) / 255.0
    out = canvas.astype(np.float32)
    color_f = np.array(fill_color, dtype=np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - alpha) + color_f[c] * alpha

    # 8. Outline
    if outline_thickness > 0:
        contours, _ = cv2.findContours(
            (mask > 127).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, outline_color, outline_thickness, cv2.LINE_AA)

    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_smpl_shadow(
    canvas: np.ndarray,
    xy: np.ndarray,
    w: int,
    h: int,
) -> None:
    """Draw a soft elliptical ground shadow from projected vertices."""
    # Use lowest vertices (highest row values = closest to ground)
    rows = xy[:, 1]
    ground_y = float(np.percentile(rows, 98))  # near-bottom vertices
    cx = int(np.clip(np.mean(xy[:, 0]), 0, w - 1))
    cy = int(np.clip(ground_y + 8, 0, h - 1))

    x_extent = float(np.percentile(xy[:, 0], 95) - np.percentile(xy[:, 0], 5))
    spread = max(20, int(x_extent * 0.5))

    shadow = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(shadow, (cx, cy), (spread, max(6, spread // 4)), 0, 0, 360, (0.45,), -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(4, spread * 0.3))
    canvas_f = canvas.astype(np.float32)
    canvas_f *= (1.0 - shadow[:, :, np.newaxis] * 0.5)
    np.clip(canvas_f, 0, 255, out=canvas_f)
    canvas[:] = canvas_f.astype(np.uint8)


# ── Availability check ──────────────────────────────────────────────────────

def is_smpl_available(gender: str = "neutral") -> bool:
    """Return ``True`` if an SMPL model file can be found.

    Searches :data:`_SEARCH_DIRS` for ``SMPL_{GENDER}.npz`` or ``.pkl``.
    """
    return _find_model_file(gender) is not None

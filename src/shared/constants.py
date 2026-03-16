
# ── Physics ──────────────────────────────────────────────────────────────

GRAVITY: float = -9.81          # m/s², PyBullet Z-down convention (PHYSICAL CONSTANT)
# ARCH CONSTRAINT: 240Hz is the PyBullet recommended simulation rate.
# Paper §3: "240 simulation steps per second". Lower → joint instability;
# higher → diminishing returns with proportional CPU cost.
DEFAULT_PHYSICS_HZ: int = 240

# ── Rendering ────────────────────────────────────────────────────────────

# STANDARD: 24fps = cinema standard (H.264). Paper §3: "24 fps output".
DEFAULT_FPS: int = 24
# DESIGN CHOICE: 5s default duration. Covers most single-action motions
# (walk cycle ~2s, jump ~1s). Not architectural.
DEFAULT_DURATION: float = 5.0
# DESIGN CHOICE: 640×480 = VGA, low-VRAM friendly for development/preview.
# Final renders can use higher resolution. Not architectural.
DEFAULT_VIDEO_WIDTH:  int = 640
DEFAULT_VIDEO_HEIGHT: int = 480

# ── Data ─────────────────────────────────────────────────────────────────

# NOTE: mirrors config/default.yaml::data.amass_dir.  Keep in sync.
DEFAULT_DATA_DIR = "data/AMASS"
ARCTIC_DATA_DIR  = "data/ARCTIC/unpack"
PAHOI_DATA_DIR   = "data/PAHOI"
INTERX_DATA_DIR  = "data/inter-x"

# ── Checkpoints ──────────────────────────────────────────────────────────

# understanding  (M1 T5 scene parser)
DEFAULT_UNDERSTANDING_CHECKPOINT = "checkpoints/understanding/scene_extractor_v5"

# motion  (MotionSSM + PhysicsSSM)
DEFAULT_PHYSICS_SSM_CHECKPOINT = "checkpoints/physics_ssm/best_model.pt"
DEFAULT_MOTION_SSM_CHECKPOINT  = "checkpoints/motion_ssm/best_model.pt"

# ── SMPL-X body model ───────────────────────────────────────────────────

# ARCH CONSTRAINTS — defined by SMPL-X body model specification:
N_JOINTS: int = 55             # SMPL-X total: 1 root + 21 body + 30 hand + 1 jaw + 2 eyes
N_BODY_JOINTS: int = 22        # pelvis + 21 body joints
N_HAND_JOINTS: int = 15        # per hand (30 total)
SMPLX_N_VERTS: int = 10475     # SMPL-X mesh vertices (fixed by model topology)

# ── Motion model hyper-parameters (used by motion module) ─────────────────

# ARCH CONSTRAINT: SMPL-X pose params = root_orient(3) + trans(3) + body(63)
# + hands(90) + jaw(3) + eyes(6) = 168.  Fixed by SMPL-X model specification.
# Datasets: AMASS, InterX, PAHOI, ARCTIC — all stored as 168-dim SMPL-X params.
MOTION_DIM:    int = 168
# STANDARD: AMASS / InterX / PAHOI / ARCTIC are captured at 30 fps.
# This is the canonical motion frame rate for generation and retrieval.
MOTION_FPS:    int = 30

# DESIGN CHOICE: 64-dim physics state vector. Paper §4: "64 dimensions,
# 24 active" (position, velocity, contact forces per joint). Sized to fit
# alongside 168-dim motion in a single SSM forward pass. Not bound by SMPL-X.
D_PHYSICS: int = 64

# ARCH CONSTRAINTS — Mamba SSM hyperparameters (paper §3):
# d_model=256: hidden size. Paper §3 Table 1.
# d_state=32: Mamba state expansion factor. Paper §3.
# n_layers=4: number of stacked Mamba blocks. Paper §3 Table 1.
SSM_D_MODEL:   int = 256
SSM_D_STATE:   int = 32
SSM_N_LAYERS:  int = 4

# ── SMPL-X 55-joint kinematic tree ──────────────────────────────────────
# parent[j] = parent joint index (-1 for root)

SMPLX_PARENTS: list[int] = [
    -1,  #  0 pelvis
     0,  #  1 left_hip
     0,  #  2 right_hip
     0,  #  3 spine1
     1,  #  4 left_knee
     2,  #  5 right_knee
     3,  #  6 spine2
     4,  #  7 left_ankle
     5,  #  8 right_ankle
     6,  #  9 spine3
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
    # --- left hand (22-36) ---
    20,  # 22 L_index1
    22,  # 23 L_index2
    23,  # 24 L_index3
    20,  # 25 L_middle1
    25,  # 26 L_middle2
    26,  # 27 L_middle3
    20,  # 28 L_pinky1
    28,  # 29 L_pinky2
    29,  # 30 L_pinky3
    20,  # 31 L_ring1
    31,  # 32 L_ring2
    32,  # 33 L_ring3
    20,  # 34 L_thumb1
    34,  # 35 L_thumb2
    35,  # 36 L_thumb3
    # --- right hand (37-51) ---
    21,  # 37 R_index1
    37,  # 38 R_index2
    38,  # 39 R_index3
    21,  # 40 R_middle1
    40,  # 41 R_middle2
    41,  # 42 R_middle3
    21,  # 43 R_pinky1
    43,  # 44 R_pinky2
    44,  # 45 R_pinky3
    21,  # 46 R_ring1
    46,  # 47 R_ring2
    47,  # 48 R_ring3
    21,  # 49 R_thumb1
    49,  # 50 R_thumb2
    50,  # 51 R_thumb3
    # --- jaw + eyes ---
    15,  # 52 jaw
    15,  # 53 left_eye
    15,  # 54 right_eye
]

# NOTE on hand joint naming: body joints (0-21) use full snake_case (e.g. "left_hip");
# hand joints (22-51) use the SMPL-X convention "L_/R_" prefix + digit suffix
# (e.g. "L_index1").  This reflects the upstream SMPL-X model definition and
# is intentionally different — do not "fix" to be consistent.
SMPLX_JOINT_NAMES: list[str] = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "L_index1", "L_index2", "L_index3", "L_middle1", "L_middle2", "L_middle3",
    "L_pinky1", "L_pinky2", "L_pinky3", "L_ring1", "L_ring2", "L_ring3",
    "L_thumb1", "L_thumb2", "L_thumb3",
    "R_index1", "R_index2", "R_index3", "R_middle1", "R_middle2", "R_middle3",
    "R_pinky1", "R_pinky2", "R_pinky3", "R_ring1", "R_ring2", "R_ring3",
    "R_thumb1", "R_thumb2", "R_thumb3",
    "jaw", "left_eye", "right_eye",
]

# Skeleton bone connectivity for visualization (body only — 22 joints)
SMPLX_BODY_BONES: list[tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3),                  # pelvis → hips + spine
    (1, 4), (2, 5), (3, 6),                  # knees + spine2
    (4, 7), (5, 8), (6, 9),                  # ankles + spine3
    (7, 10), (8, 11),                         # feet
    (9, 12), (9, 13), (9, 14),               # neck + collars
    (12, 15),                                  # head
    (13, 16), (14, 17),                       # shoulders
    (16, 18), (17, 19),                       # elbows
    (18, 20), (19, 21),                       # wrists
]

# ── 21-joint humanoid skeleton bone connectivity (PyBullet humanoid.urdf) ────

HUMANOID_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # spine + head
    (2, 5), (5, 6), (6, 7),                   # left arm
    (2, 8), (8, 9), (9, 10),                  # right arm
    (0, 11), (0, 16), (11, 16),               # pelvis → hips
    (11, 12), (12, 13), (13, 14), (14, 15),   # left leg
    (16, 17), (17, 18), (18, 19), (19, 20),   # right leg
    (5, 8),                                    # shoulder girdle
]

# ── Colour palette (RGBA float, used by M1 parsers) ─────────────────────

SCENE_COLORS: dict[str, tuple[float, float, float, float]] = {
    "red":    (1.0, 0.1, 0.1, 1.0), "green":  (0.1, 0.8, 0.1, 1.0),
    "blue":   (0.1, 0.1, 1.0, 1.0), "yellow": (1.0, 0.9, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0, 1.0), "purple": (0.5, 0.0, 0.8, 1.0),
    "white":  (0.95, 0.95, 0.95, 1.0), "black": (0.1, 0.1, 0.1, 1.0),
    "gray":   (0.5, 0.5, 0.5, 1.0), "grey":   (0.5, 0.5, 0.5, 1.0),
    "brown":  (0.5, 0.3, 0.1, 1.0), "pink":   (1.0, 0.5, 0.7, 1.0),
    "cyan":   (0.0, 0.8, 0.8, 1.0),
}

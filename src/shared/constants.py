"""
#WHERE
    Imported by pipeline.py and all modules — single source of truth
    for all constants, paths, and default hyper-parameters.

#WHAT
    Centralised constants used across 3+ modules.  Edit here, not in
    individual module files.

#INPUT / #OUTPUT
    Pure constants — no I/O.
"""

# ── Physics ──────────────────────────────────────────────────────────────

GRAVITY: float = -9.81          # m/s², PyBullet Z-down convention
DEFAULT_PHYSICS_HZ: int = 240   # simulation steps per second

# ── Rendering ────────────────────────────────────────────────────────────

DEFAULT_FPS: int = 24           # output video frame rate
DEFAULT_DURATION: float = 6.0   # seconds
DEFAULT_VIDEO_WIDTH:  int = 640
DEFAULT_VIDEO_HEIGHT: int = 480

# ── Data ─────────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = "data/KIT-ML"

# ── Checkpoints ──────────────────────────────────────────────────────────

# scene_understanding  (M1 T5 scene parser)
DEFAULT_SCENE_UNDERSTANDING_CHECKPOINT = "checkpoints/scene_understanding/scene_extractor_v5"

# motion_generator  (MotionSSM + PhysicsSSM)
DEFAULT_PHYSICS_SSM_CHECKPOINT = "checkpoints/physics_ssm/best_model.pt"
DEFAULT_MOTION_SSM_CHECKPOINT  = "checkpoints/motion_ssm/best_model.pt"

# ── Motion model hyper-parameters (used by motion_generator) ─────────────

MOTION_DIM:    int = 251   # KIT-ML / HumanML3D feature dimensionality
MOTION_FPS:    int = 20    # KIT-ML capture frame rate

SSM_D_MODEL:   int = 256
SSM_D_STATE:   int = 32
SSM_N_LAYERS:  int = 4

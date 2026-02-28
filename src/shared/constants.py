"""
#WHERE
    Imported by pipeline.py, M1, M4, M5, M7, M8 — any module needing
    physics or rendering constants.

#WHAT
    Centralised physical and rendering constants used across 3+ modules.
    Single source of truth — no more magic numbers scattered in code.

#INPUT / #OUTPUT
    Pure constants — no I/O.
"""

# ── Physics ──────────────────────────────────────────────────────────────

GRAVITY: float = -9.81          # m/s², PyBullet Z-down convention
DEFAULT_PHYSICS_HZ: int = 240   # simulation steps per second

# ── Rendering ────────────────────────────────────────────────────────────

DEFAULT_FPS: int = 24           # output video frame rate
DEFAULT_DURATION: float = 6.0   # seconds

# ── Paths ────────────────────────────────────────────────────────────────

DEFAULT_PHYSICS_SSM_CHECKPOINT = "checkpoints/physics_ssm/best_model.pt"
DEFAULT_MOTION_SSM_CHECKPOINT  = "checkpoints/motion_ssm/best_model.pt"

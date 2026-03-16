"""Backward-compatibility shim  moved to src.shared.data.physics_dataset."""
from src.shared.data.physics_dataset import (  # noqa: F401
    PhysicsMotionDataset,
    extract_physics_state,
    compute_global_physics_stats,
    normalize_physics,
)

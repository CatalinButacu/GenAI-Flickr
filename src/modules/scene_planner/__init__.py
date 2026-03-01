"""
#WHERE
    Imported by pipeline.py, benchmarks, test_modules.py.

#WHAT
    Scene Planner Module (Module 2) — plans spatial layout of scene entities.
    Constraint-based solver (scipy) or row-based fallback.

#INPUT
    ParsedScene from M1, or dict with entities + relations.

#OUTPUT
    PlannedScene with positioned PlannedEntity list.
"""

from .models import Position3D, PlannedEntity, PlannedScene
from .planner import ScenePlanner, plan_scene
from .constraint_layout import solve_layout, build_constraints, SpatialConstraint

__all__ = [
    "ScenePlanner",
    "PlannedScene",
    "PlannedEntity",
    "Position3D",
    "plan_scene",
    "solve_layout",
    "build_constraints",
    "SpatialConstraint",
]

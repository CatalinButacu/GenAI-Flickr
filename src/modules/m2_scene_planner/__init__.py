"""
Scene Planner Module (Module 2)
===============================
Plans spatial layout of scene entities before physics simulation.

Two placement modes:
  1. Constraint-based: scipy.optimize solver for relation triples (M1 T5 output)
  2. Row-based fallback: hard-coded constants for simple scenes

Input: ParsedScene from Module 1, or dict from M1 T5 model
Output: PlannedScene with positioned entities

Example:
    from src.modules.m2_scene_planner import ScenePlanner, plan_scene
    
    # From M1 T5 dict:
    planned = plan_scene({"entities": [...], "relations": [...]})
    
    # From ParsedScene:
    planned = plan_scene(parsed_scene)
"""

from .planner import (
    ScenePlanner,
    PlannedScene,
    PlannedEntity,
    Position3D,
    plan_scene,
)
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

"""
Scene Planner Module (Module 2)
===============================
Plans spatial layout of scene entities before physics simulation.

Input: ParsedScene from Module 1
Output: PlannedScene with positioned entities

Example:
    from src.modules.m2_scene_planner import ScenePlanner, plan_scene
    
    planned = plan_scene(parsed_scene)
    for entity in planned.entities:
        print(f"{entity.name} at {entity.position}")
"""

from .planner import (
    ScenePlanner,
    PlannedScene,
    PlannedEntity,
    Position3D,
    plan_scene,
)

__all__ = [
    "ScenePlanner",
    "PlannedScene",
    "PlannedEntity",
    "Position3D",
    "plan_scene",
]

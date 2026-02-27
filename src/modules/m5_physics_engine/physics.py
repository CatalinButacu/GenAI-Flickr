"""
#WHERE
    Convenience re-export — used by test scripts and demo scripts.

#WHAT
    M5 Physics Engine re-exports and vocabulary-aware factory helpers.

#INPUT
    Gravity constant (default -9.81).

#OUTPUT
    Configured Scene, factory helper functions.
"""

# Re-export existing physics components
from .scene import Scene, PhysicsObject
from .simulator import Simulator, FrameData, CameraConfig, CinematicCamera

# Import vocabulary for type hints
from src.shared.vocabulary import ObjectDefinition, OBJECTS


def create_physics_scene(gravity: float = -9.81) -> Scene:
    scene = Scene(gravity=gravity)
    scene.setup()
    return scene


def add_object_from_vocabulary(
    scene: Scene,
    obj_def: ObjectDefinition,
    name: str,
    position: list = None,
    color: tuple = None
) -> PhysicsObject:
    """Bridge M1 vocabulary output to a PyBullet physics object."""
    position = position or [0, 0, 0.5]
    color    = list(color) if color else obj_def.default_size
    if obj_def.category.name == "HUMANOID":
        raise NotImplementedError("Humanoid bodies handled by humanoid.py")
    return scene.add_primitive(
        name=name, shape=obj_def.default_shape, size=obj_def.default_size,
        mass=obj_def.default_mass, position=position, color=color,
        is_static=(obj_def.default_mass == 0),
    )


__all__ = [
    "Scene",
    "PhysicsObject",
    "Simulator",
    "FrameData",
    "CameraConfig",
    "CinematicCamera",
    "create_physics_scene",
    "add_object_from_vocabulary",
]

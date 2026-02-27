"""
#WHERE
    Used by planner.py, constraint_layout.py, pipeline.py, m5_physics_engine,
    benchmarks (benchmark_m2.py), and test_modules.py.

#WHAT
    Spatial data models for the scene planner: 3D position vector,
    positioned entity with physics attributes, and planned scene container.

#INPUT
    Coordinates, entity metadata (name, type, size, mass, color).

#OUTPUT
    Position3D, PlannedEntity, PlannedScene dataclass instances.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(slots=True)
class Position3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def __add__(self, other: "Position3D") -> "Position3D":
        return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)


@dataclass(slots=True)
class PlannedEntity:
    name: str
    object_type: str
    position: Position3D
    rotation: tuple[float, float, float, float] = (0, 0, 0, 1)
    color: Optional[tuple[float, ...]] = None
    is_actor: bool = False
    size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0


@dataclass(slots=True)
class PlannedScene:
    entities: list[PlannedEntity]
    ground_size: tuple[float, float] = (10.0, 10.0)
    camera_distance: float = 2.5

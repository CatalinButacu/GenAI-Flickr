from __future__ import annotations

from dataclasses import dataclass, field

from src.modules.physics.body_model import BodyParams


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
    color: tuple[float, ...] | None = None
    is_actor: bool = False
    size: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    mass: float = 1.0
    body_params: BodyParams | None = None


@dataclass(slots=True)
class PlannedScene:
    entities: list[PlannedEntity]
    ground_size: tuple[float, float] = (10.0, 10.0)
    camera_distance: float = 2.5

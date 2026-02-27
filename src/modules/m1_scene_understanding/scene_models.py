"""
#WHERE
    Used by builder.py, orchestrator.py, pipeline.py (the main pipeline),
    and test files that verify end-to-end scene output.

#WHAT
    Data models for the final scene description that M1 outputs.
    These are consumed by M2 (ScenePlanner) and M5 (PhysicsEngine).

#INPUT
    Structured fields: object shapes, positions, colours, camera motions, etc.

#OUTPUT
    SceneDescription containing lists of SceneObject, SceneAction, CameraMotion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class SceneObject:
    name: str
    shape: str = "box"
    size: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.5])
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    mass: float = 1.0
    is_static: bool = False
    mesh_prompt: Optional[str] = None
    material: str = "default"


@dataclass(slots=True)
class SceneAction:
    time: float
    object_name: str
    action_type: str = "force"
    vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass(slots=True)
class CameraMotion:
    motion_type: str = "orbit"
    start_time: float = 0.0
    duration: float = 5.0
    start_value: float = 0.0
    end_value: float = 360.0


@dataclass
class SceneDescription:
    prompt: str = ""
    objects: List[SceneObject] = field(default_factory=list)
    actions: List[SceneAction] = field(default_factory=list)
    camera_motions: List[CameraMotion] = field(default_factory=list)
    duration: float = 5.0
    fps: int = 24
    style_prompt: str = "realistic, cinematic lighting, high quality"
    environment: str = "default"

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def has_actions(self) -> bool:
        return bool(self.actions)

from __future__ import annotations

import random
from collections import deque
from dataclasses import replace
from typing import Dict, List, Tuple

from .models import (
    EntityType, ExtractedRelation, ExtractionResult,
    SceneDescription, SceneObject, SceneAction, CameraMotion,
)

# 3D spatial offset per relation — geometry
RELATION_OFFSETS: Dict[str, Tuple[float, float, float]] = {
    "in_front_of": (0.0, 0.0, -1.0),
    "behind":      (0.0, 0.0,  1.0),
    "left_of":     (-1.0, 0.0, 0.0),
    "right_of":    (1.0,  0.0, 0.0),
    "beside":      (1.0,  0.0, 0.0),
    "on_top_of":   (0.0,  1.0, 0.0),
    "under":       (0.0, -1.0, 0.0),
    "inside":      (0.0,  0.0, 0.0),
    "facing":      (0.0,  0.0, -1.0),
    "near":        (0.8,  0.0, 0.0),
}

# Symmetric/inverse relation pairs — geometric identity
INVERSE_RELATIONS: Dict[str, str] = {
    "in_front_of": "behind",   "behind":    "in_front_of",
    "left_of":     "right_of", "right_of":  "left_of",
    "on_top_of":   "under",    "under":     "on_top_of",
    "inside":      "inside",   "facing":    "facing",
    "beside":      "beside",   "near":      "near",
}

# Physics engine constants per material — engineering values
MATERIAL_PHYSICS: Dict[str, Dict[str, float]] = {
    "metal":    {"friction": 0.4, "restitution": 0.3},
    "wood":     {"friction": 0.5, "restitution": 0.2},
    "plastic":  {"friction": 0.35,"restitution": 0.4},
    "rubber":   {"friction": 0.8, "restitution": 0.7},
    "glass":    {"friction": 0.3, "restitution": 0.1},
    "stone":    {"friction": 0.6, "restitution": 0.1},
    "concrete": {"friction": 0.6, "restitution": 0.1},
    "fabric":   {"friction": 0.7, "restitution": 0.05},
    "default":  {"friction": 0.5, "restitution": 0.3},
}

_ENTITY_CATEGORY = {
    EntityType.PERSON:      "person",
    EntityType.ANIMAL:      "animal",
    EntityType.VEHICLE:     "vehicle",
    EntityType.OBJECT:      "object",
    EntityType.ENVIRONMENT: "environment",
}


class SceneBuilder:
    """Positions extracted entities in 3D space and emits a SceneDescription."""

    def __init__(self, spacing: float = 2.0, jitter: float = 0.15) -> None:
        self._spacing = spacing
        self._jitter  = jitter

    @property
    def spacing(self) -> float:
        return self._spacing

    @spacing.setter
    def spacing(self, value: float) -> None:
        if value <= 0:
            raise ValueError("spacing must be positive")
        self._spacing = value

    @property
    def jitter(self) -> float:
        return self._jitter

    @jitter.setter
    def jitter(self, value: float) -> None:
        if value < 0:
            raise ValueError("jitter must be non-negative")
        self._jitter = value

    def build(self, extraction: ExtractionResult) -> SceneDescription:
        objects  = self._place_objects(self._make_objects(extraction), extraction.relations)
        actions  = self._make_actions(extraction, {o.name for o in objects})
        camera   = self._make_camera(objects, actions)
        duration = max(3.0, min(sum(a.time + 2.0 for a in actions), 15.0)) if actions else 5.0

        return SceneDescription(
            prompt=extraction.raw_prompt,
            objects=objects,
            actions=actions,
            camera_motions=[camera] if camera else [],
            duration=duration,
            style_prompt=extraction.style_prompt,
            environment=extraction.inferred_setting or "default",
        )

    # -- private --

    def _make_objects(self, extraction: ExtractionResult) -> List[SceneObject]:
        result = []
        for e in extraction.entities:
            cat   = _ENTITY_CATEGORY.get(e.entity_type, "object")
            phys  = MATERIAL_PHYSICS.get((e.material or "default").lower(), MATERIAL_PHYSICS["default"])
            size  = [e.dimensions.get(k, 0.1) for k in ("length", "width", "height")] if e.dimensions else [0.1, 0.1, 0.1]
            color = e.color or [0.5, 0.5, 0.5, 1.0]

            for i in range(max(e.count, 1)):
                obj_id = f"{e.id}_{i}" if e.count > 1 else e.id
                result.append(SceneObject(
                    name=obj_id,
                    shape="humanoid" if cat == "person" else "box",
                    size=size,
                    position=[0.0, 0.0, 0.5],
                    color=color,
                    mass=e.mass or 1.0,
                    is_static=bool(e.is_static),
                    mesh_prompt=e.mesh_prompt or f"a {e.name}",
                    material=e.material or "default",
                ))
        return result

    def _place_objects(self, objects: List[SceneObject], relations: List[ExtractedRelation]) -> List[SceneObject]:
        if not objects:
            return objects

        id_map = {o.name: i for i, o in enumerate(objects)}
        placed: set = set()
        result = list(objects)

        # Build adjacency from relations
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for r in relations:
            if r.source_id in id_map and r.target_id in id_map:
                adj.setdefault(r.source_id, []).append((r.target_id, r.relation))
                adj.setdefault(r.target_id, []).append((r.source_id, INVERSE_RELATIONS.get(r.relation, r.relation)))

        # BFS layout from first object
        root = objects[0].name
        result[id_map[root]] = replace(result[id_map[root]], position=[0.0, 0.0, 0.5])
        placed.add(root)
        queue = deque([root])

        while queue:
            cur = queue.popleft()
            cx, cy, cz = result[id_map[cur]].position
            for neighbor, rel in adj.get(cur, []):
                if neighbor in placed:
                    continue
                dx, dy, dz = RELATION_OFFSETS.get(rel, (1.0, 0.0, 0.0))
                pos = [
                    cx + dx * self._spacing + random.uniform(-self._jitter, self._jitter),
                    cy + dy * self._spacing,
                    cz + dz * self._spacing + random.uniform(-self._jitter, self._jitter),
                ]
                result[id_map[neighbor]] = replace(result[id_map[neighbor]], position=pos)
                placed.add(neighbor)
                queue.append(neighbor)

        # Place any unlinked objects along x-axis
        x = self._spacing
        for i, obj in enumerate(result):
            if obj.name not in placed:
                result[i] = replace(obj, position=[x, 0.0, 0.5])
                x += self._spacing

        return result

    def _make_actions(self, extraction: ExtractionResult, object_names: set) -> List[SceneAction]:
        return [
            SceneAction(
                time=0.0,
                object_name=a.actor_id,
                action_type=a.parameters.get("action_type", "force"),
                vector=a.parameters.get("vector", [0.0, 0.0, 0.0]),
            )
            for a in extraction.actions
            if a.actor_id in object_names
        ]

    def _make_camera(self, objects: List[SceneObject], actions: List[SceneAction]) -> CameraMotion | None:
        if not objects:
            return None
        duration = max(3.0, min(sum(a.time + 2.0 for a in actions), 15.0)) if actions else 5.0
        return CameraMotion(motion_type="orbit", start_value=0.0, end_value=360.0, duration=duration)

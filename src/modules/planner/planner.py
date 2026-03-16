from __future__ import annotations

import logging
from typing import Any

from src.shared.vocabulary import OBJECTS
from .models import Position3D, PlannedEntity, PlannedScene  # noqa: F401 — re-export
from src.modules.planner.constraint_layout import solve_layout

log = logging.getLogger(__name__)


class ScenePlanner:
    """Computes 3D positions from parsed spatial relations and actions.

    When the input scene has relation triples (from M1 T5 model), uses
    scipy.optimize constraint solver for physically plausible layout.
    Falls back to simple row-based placement otherwise.
    """

    _ACTOR_DIST = 1.5
    _OBJ_SPACE  = 0.4
    _FALL_H     = 1.5
    _GROUND_H   = 0.5
    _CLOSE_ACT  = {"kick": 0.8, "pick_up": 0.5}

    def plan(self, parsed_scene) -> PlannedScene:
        """Plan entity positions from ParsedScene or dict."""
        if isinstance(parsed_scene, dict):
            return self._plan_dict(parsed_scene)
        return self._plan_parsed(parsed_scene)

    def _plan_dict(self, scene_dict: dict[str, Any]) -> PlannedScene:
        """Plan layout from M1 T5 model output (slim JSON dict)."""
        entities_raw = scene_dict.get("entities", [])
        relations = scene_dict.get("relations", [])
        entity_names = [e.get("name", f"obj_{i}") for i, e in enumerate(entities_raw)]
        solved = self.try_constraint_solve(entity_names, relations)

        planned: list[PlannedEntity] = []
        for i, e in enumerate(entities_raw):
            name = e.get("name", f"obj_{i}")
            etype = e.get("type", "object")
            pos = Position3D(*solved[name]) if solved and name in solved else Position3D(i * self._OBJ_SPACE, 0, self._GROUND_H)
            od = OBJECTS.get(etype)
            planned.append(PlannedEntity(
                name=name, object_type=etype, position=pos,
                size=od.default_size if od else [0.1, 0.1, 0.1],
                mass=od.default_mass if od else 1.0,
            ))

        mode = "constraint" if solved else "row"
        log.info("ScenePlanner (%s): %d entities positioned", mode, len(planned))
        return PlannedScene(entities=planned)

    def _plan_parsed(self, parsed_scene) -> PlannedScene:
        """Plan layout from ParsedScene."""
        relations = getattr(parsed_scene, "spatial_relations", [])
        has_triples = (relations and isinstance(relations[0], dict)
                       and "subject" in relations[0])

        if has_triples:
            entity_names = [e.name for e in parsed_scene.entities]
            if solved := self.try_constraint_solve(entity_names, relations):
                return self._build_planned(parsed_scene.entities, solved, "constraint")

        # Fallback: row-based placement
        actors  = [e for e in parsed_scene.entities if e.is_actor]
        objects = [e for e in parsed_scene.entities if not e.is_actor]
        obj_pos = self.place_objects(objects, parsed_scene.actions)
        act_pos = self.place_actors(actors, parsed_scene.actions, obj_pos)
        pos_map = {**obj_pos, **act_pos}
        return self._build_planned(parsed_scene.entities, pos_map, "row")

    def _build_planned(self, entities, pos_map, mode: str) -> PlannedScene:
        """Build PlannedScene from entity list + position map."""
        planned: list[PlannedEntity] = []
        for e in entities:
            raw = pos_map.get(e.name) if hasattr(e, 'name') else None
            if isinstance(raw, Position3D):
                pos = raw
            elif isinstance(raw, tuple):
                pos = Position3D(*raw)
            else:
                pos = Position3D(0, -self._ACTOR_DIST if getattr(e, 'is_actor', False) else 0, self._GROUND_H)
            od = OBJECTS.get(getattr(e, 'object_type', 'object'))
            planned.append(PlannedEntity(
                name=e.name, object_type=getattr(e, 'object_type', 'object'),
                position=pos,
                color=getattr(e, 'color', None),
                is_actor=getattr(e, 'is_actor', False),
                size=od.default_size if od else [0.1, 0.1, 0.1],
                mass=od.default_mass if od else 1.0,
                body_params=getattr(e, 'body_params', None),
            ))
        log.info("ScenePlanner (%s): %d entities positioned", mode, len(planned))
        return PlannedScene(entities=planned)

    @staticmethod
    def try_constraint_solve(
        entity_names: list[str], relations: list[dict]
    ) -> dict[str, tuple]:
        """Attempt constraint-based layout. Returns empty dict on failure/no constraints."""
        try:
            return solve_layout(entity_names, relations)
        except Exception as exc:
            log.warning("Constraint solver failed: %s — falling back", exc)
            return {}

    def place_objects(self, objects, actions) -> dict[str, Position3D]:
        pos: dict[str, Position3D] = {}
        for i, obj in enumerate(objects):
            z = self.resolve_object_z(obj.name, actions)
            pos[obj.name] = Position3D(i * self._OBJ_SPACE, 0, z)
        self.apply_two_object_fall(objects, actions, pos)
        return pos

    def resolve_object_z(self, name: str, actions) -> float:
        """Return ground or fall height based on whether the object is falling."""
        for a in actions:
            if a.action_type == "fall":
                return self._FALL_H if a.actor == name else self._GROUND_H
        return self._GROUND_H

    def apply_two_object_fall(self, objects, actions, pos: dict[str, Position3D]):
        """Override positions when exactly two objects participate in a fall."""
        if len(objects) != 2:
            return
        for a in actions:
            if a.action_type != "fall" or not a.target:
                continue
            if falling := next((o.name for o in objects if o.name != a.target), None):
                pos[a.target] = Position3D(0, 0, 0.1)
                pos[falling]  = Position3D(0, 0, self._FALL_H)
            break

    def place_actors(self, actors, actions, obj_pos) -> dict[str, Position3D]:
        pos: dict[str, Position3D] = {}
        for actor in actors:
            target, act_type = None, None
            for a in actions:
                if a.actor == actor.name and a.target:
                    target, act_type = a.target, a.action_type
                    break
            if target and target in obj_pos:
                tp = obj_pos[target]
                d  = self._CLOSE_ACT.get(act_type, self._ACTOR_DIST)
                pos[actor.name] = Position3D(tp.x, tp.y - d, 0)
            else:
                pos[actor.name] = Position3D(0, -self._ACTOR_DIST, 0)
        return pos


def plan_scene(parsed_scene) -> PlannedScene:
    return ScenePlanner().plan(parsed_scene)

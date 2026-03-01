"""
#WHERE
    Used by pipeline.py, benchmarks (benchmark_m2.py), test_modules.py.
    Re-exports Position3D, PlannedEntity, PlannedScene for backward compat.

#WHAT
    Scene planner — computes 3D positions from parsed spatial relations.
    Two modes: constraint-based (scipy.optimize) or row-based fallback.

#INPUT
    ParsedScene (from M1 PromptParser/T5) or dict with entities + relations.

#OUTPUT
    PlannedScene with positioned PlannedEntity list.
"""

import logging
from typing import Any, Dict, List

from src.shared.vocabulary import OBJECTS
from .models import Position3D, PlannedEntity, PlannedScene  # noqa: F401 — re-export

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
        """Plan entity positions.

        Accepts either:
          - ParsedScene (from PromptParser, with .entities, .actions, .spatial_relations)
          - dict (from M1 T5 output, with "entities" and "relations" keys)
        """
        # ── Normalize input ──────────────────────────────────────────
        if isinstance(parsed_scene, dict):
            return self._plan_from_dict(parsed_scene)
        return self._plan_from_parsed(parsed_scene)

    def _plan_from_dict(self, scene_dict: Dict[str, Any]) -> PlannedScene:
        """Plan layout from M1 T5 model output (slim JSON dict)."""
        entities_raw = scene_dict.get("entities", [])
        relations = scene_dict.get("relations", [])
        entity_names = [e.get("name", f"obj_{i}") for i, e in enumerate(entities_raw)]

        # Try constraint solver
        solved = self._try_constraint_solve(entity_names, relations)

        planned: List[PlannedEntity] = []
        for i, e in enumerate(entities_raw):
            name = e.get("name", f"obj_{i}")
            etype = e.get("type", "object")
            if solved and name in solved:
                x, y, z = solved[name]
                pos = Position3D(x, y, z)
            else:
                pos = Position3D(i * self._OBJ_SPACE, 0, self._GROUND_H)

            od = OBJECTS.get(etype)
            planned.append(PlannedEntity(
                name=name, object_type=etype, position=pos,
                size=od.default_size if od else [0.1, 0.1, 0.1],
                mass=od.default_mass if od else 1.0,
            ))

        mode = "constraint" if solved else "row"
        log.info("ScenePlanner (%s): %d entities positioned", mode, len(planned))
        return PlannedScene(entities=planned)

    def _plan_from_parsed(self, parsed_scene) -> PlannedScene:
        """Plan layout from ParsedScene (PromptParser or StoryAgent output)."""
        # Check if spatial_relations have subject/object triples (enriched format)
        relations = getattr(parsed_scene, "spatial_relations", [])
        has_triples = (relations and isinstance(relations[0], dict)
                       and "subject" in relations[0])

        if has_triples:
            entity_names = [e.name for e in parsed_scene.entities]
            solved = self._try_constraint_solve(entity_names, relations)
            if solved:
                return self._build_scene_from_solved(parsed_scene, solved)

        # Fallback: original row-based placement
        actors  = [e for e in parsed_scene.entities if e.is_actor]
        objects = [e for e in parsed_scene.entities if not e.is_actor]
        obj_pos = self._place_objects(objects, parsed_scene.actions)
        act_pos = self._place_actors(actors, parsed_scene.actions, obj_pos)
        entities: List[PlannedEntity] = []
        for e in parsed_scene.entities:
            pos = (act_pos if e.is_actor else obj_pos).get(
                e.name, Position3D(0, -self._ACTOR_DIST if e.is_actor else 0, self._GROUND_H)
            )
            od = OBJECTS.get(e.object_type)
            entities.append(PlannedEntity(
                name=e.name, object_type=e.object_type, position=pos,
                color=e.color, is_actor=e.is_actor,
                size=od.default_size if od else [0.1, 0.1, 0.1],
                mass=od.default_mass if od else 1.0,
            ))
        log.info("ScenePlanner (row): %d entities positioned", len(entities))
        return PlannedScene(entities=entities)

    def _build_scene_from_solved(self, parsed_scene, solved: Dict) -> PlannedScene:
        """Build PlannedScene from constraint-solved positions + ParsedScene metadata."""
        entities: List[PlannedEntity] = []
        for e in parsed_scene.entities:
            if e.name in solved:
                x, y, z = solved[e.name]
                pos = Position3D(x, y, z)
            else:
                pos = Position3D(0, 0, self._GROUND_H)
            od = OBJECTS.get(e.object_type)
            entities.append(PlannedEntity(
                name=e.name, object_type=e.object_type, position=pos,
                color=e.color, is_actor=e.is_actor,
                size=od.default_size if od else [0.1, 0.1, 0.1],
                mass=od.default_mass if od else 1.0,
            ))
        log.info("ScenePlanner (constraint): %d entities positioned", len(entities))
        return PlannedScene(entities=entities)

    @staticmethod
    def _try_constraint_solve(
        entity_names: List[str], relations: List[Dict]
    ) -> Dict[str, tuple]:
        """Attempt constraint-based layout. Returns empty dict on failure/no constraints."""
        from src.modules.scene_planner.constraint_layout import solve_layout
        try:
            return solve_layout(entity_names, relations)
        except Exception as exc:
            log.warning("Constraint solver failed: %s — falling back", exc)
            return {}

    def _place_objects(self, objects, actions) -> Dict[str, Position3D]:
        pos: Dict[str, Position3D] = {}
        for i, obj in enumerate(objects):
            z = self._resolve_object_z(obj.name, actions)
            pos[obj.name] = Position3D(i * self._OBJ_SPACE, 0, z)
        self._apply_two_object_fall(objects, actions, pos)
        return pos

    def _resolve_object_z(self, name: str, actions) -> float:
        """Return ground or fall height based on whether the object is falling."""
        for a in actions:
            if a.action_type == "fall":
                return self._FALL_H if a.actor == name else self._GROUND_H
        return self._GROUND_H

    def _apply_two_object_fall(self, objects, actions, pos: Dict[str, Position3D]):
        """Override positions when exactly two objects participate in a fall."""
        if len(objects) != 2:
            return
        for a in actions:
            if a.action_type != "fall" or not a.target:
                continue
            falling = next((o.name for o in objects if o.name != a.target), None)
            if falling:
                pos[a.target] = Position3D(0, 0, 0.1)
                pos[falling]  = Position3D(0, 0, self._FALL_H)
            break

    def _place_actors(self, actors, actions, obj_pos) -> Dict[str, Position3D]:
        pos: Dict[str, Position3D] = {}
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

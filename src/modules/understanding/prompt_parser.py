from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.modules.physics.body_model import BodyParams, body_params_from_prompt
from src.shared.constants import SCENE_COLORS
from src.shared.vocabulary import ACTIONS, OBJECTS, ActionCategory, ObjectCategory

log = logging.getLogger(__name__)

# Quantifier words → count mapping for entity counting
_QUANTIFIERS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "single": 1,
    "two": 2, "pair": 2, "couple": 2,
    "three": 3, "triple": 3,
    "four": 4, "five": 5, "six": 6,
    "several": 3, "few": 3, "many": 4, "multiple": 3,
}

# Generic fallback words that should map to a default object (sphere)
_GENERIC_OBJECT_WORDS = frozenset({
    "something", "object", "thing", "stuff", "item", "it",
})


@dataclass(slots=True)
class ParsedEntity:
    name: str
    object_type: str
    is_actor: bool = False
    color: tuple[float, float, float, float] | None = None
    body_params: BodyParams | None = None
    material: str | None = None    # filled by KnowledgeRetriever if KB has entry


@dataclass(slots=True)
class ParsedAction:
    action_type: str
    actor: str = ""
    target: str = ""
    raw_text: str = ""  # full matched clause from the original prompt (preserves direction/modifier words)


@dataclass(slots=True)
class ParsedScene:
    """M1 -> M2 contract. The parser must return this type."""
    prompt: str = ""
    entities: list[ParsedEntity] = field(default_factory=list)
    actions: list[ParsedAction] = field(default_factory=list)
    spatial_relations: list[dict] = field(default_factory=list)
    duration: float = 5.0
    style_prompt: str = "realistic, cinematic lighting, high quality"


_SPATIAL: dict[str, str] = {
    "on top of": "ON", "on": "ON", "above": "ABOVE", "over": "ABOVE",
    "beneath": "UNDER", "under": "UNDER", "below": "UNDER",
    "in front of": "IN_FRONT_OF", "behind": "BEHIND",
    "beside": "BESIDE", "next to": "BESIDE", "near": "NEAR",
    "left of": "LEFT_OF", "right of": "RIGHT_OF",
    "inside": "INSIDE", "in": "INSIDE",
}

_RGBA_TO_NAME: dict[tuple, str] = {v: k for k, v in SCENE_COLORS.items()}


class PromptParser:
    """Rules-based, ML-free, GPU-free scene parser. Fast path for M1."""

    def __init__(self) -> None:
        self._obj_kw: dict[str, str] = {
            kw.lower(): key for key, d in OBJECTS.items() for kw in d.keywords
        }
        self._act_kw: dict[str, str] = {
            kw.lower(): key for key, d in ACTIONS.items() for kw in d.keywords
        }
        # Pre-compile longest-first to avoid partial keyword matches
        self._obj_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(r"\b" + re.escape(kw) + r"\b"), kw)
            for kw in sorted(self._obj_kw, key=len, reverse=True)
        ]
        self._act_patterns: list[tuple[re.Pattern, str]] = [
            (re.compile(r"\b" + re.escape(kw) + r"\b"), kw)
            for kw in sorted(self._act_kw, key=len, reverse=True)
        ]
        self._spatial_phrases = sorted(_SPATIAL, key=len, reverse=True)

    def setup(self) -> None: ...

    def parse(self, prompt: str) -> ParsedScene:
        # ── Input sanitisation ──────────────────────────────────────
        prompt = (prompt or "").strip()
        if not prompt:
            log.warning("[M1-Rules] empty prompt — returning minimal scene")
            return ParsedScene(prompt=prompt)

        log.info("[M1-Rules] parsing prompt: %r", prompt)
        text = prompt.lower()
        entities = self.extract_entities(text)
        actions = self.extract_actions(text, entities)
        spatial = self.extract_spatial(text)

        entities, actions = self._apply_fallbacks(entities, actions, text)

        log.info("[M1-Rules] found %d entities, %d actions, %d spatial relations",
                 len(entities), len(actions), len(spatial))
        self._log_parse_results(entities, actions)
        return ParsedScene(
            prompt=prompt,
            entities=entities,
            actions=actions,
            spatial_relations=spatial,
        )

    def _apply_fallbacks(self, entities, actions, text):
        """Apply fallback rules when no entities or actions detected."""
        if not entities:
            entities = self._infer_fallback_entities(text)
        if not actions and any(e.is_actor for e in entities):
            actors = [e for e in entities if e.is_actor]
            actions = [ParsedAction(
                action_type="stand", actor=actors[0].name, raw_text=text,
            )]
            log.info("[M1-Rules] no actions found — defaulting to 'stand' for %s",
                     actors[0].name)
        return entities, actions

    @staticmethod
    def _log_parse_results(entities, actions):
        """Log detected entities and actions."""
        for e in entities:
            log.info("[M1-Rules]   entity: %s (type=%s, actor=%s)",
                     e.name, e.object_type, e.is_actor)
        for a in actions:
            log.info("[M1-Rules]   action: %s (actor=%s, target=%s)",
                     a.action_type, a.actor, a.target)

    def extract_entities(self, text: str) -> list[ParsedEntity]:
        color_ctx = self.color_context(text)
        count_ctx = self._extract_quantifiers(text)
        seen: set[str] = set()
        entities: list[ParsedEntity] = []
        for pattern, kw in self._obj_patterns:
            if not pattern.search(text):
                continue
            obj_type = self._obj_kw[kw]
            if obj_type in seen:
                continue
            seen.add(obj_type)
            obj_def = OBJECTS.get(obj_type)
            is_actor = bool(obj_def and obj_def.category == ObjectCategory.HUMANOID)
            color = color_ctx.get(kw)
            prefix = _RGBA_TO_NAME.get(color) if color else ""
            count = count_ctx.get(kw, 1)
            # Extract parametric body shape for human actors
            bp = body_params_from_prompt(text, entity_name=kw) if is_actor else None

            for idx in range(count):
                if count > 1:
                    name = f"{prefix}_{obj_type}_{idx + 1}".strip("_")
                else:
                    name = f"{prefix}_{obj_type}".strip("_") if prefix else obj_type
                entities.append(ParsedEntity(
                    name=name, object_type=obj_type, is_actor=is_actor,
                    color=color, body_params=bp,
                ))
        return entities

    def _extract_quantifiers(self, text: str) -> dict[str, int]:
        """Look for 'two balls', 'three cubes', etc. and return {keyword: count}."""
        result: dict[str, int] = {}
        tokens = text.split()
        for i, token in enumerate(tokens):
            clean = re.sub(r"[^a-z]", "", token)
            if clean not in _QUANTIFIERS:
                continue
            count = _QUANTIFIERS[clean]
            # Look ahead for the next object keyword
            window = " ".join(tokens[i + 1: i + 4])
            for pattern, kw in self._obj_patterns:
                if pattern.search(window):
                    result[kw] = count
                    break
        return result

    def _infer_fallback_entities(self, text: str) -> list[ParsedEntity]:
        """When no objects were matched, try generic fallback words."""
        tokens = set(re.findall(r'\b\w+\b', text))
        if tokens & _GENERIC_OBJECT_WORDS:
            log.info("[M1-Rules] no objects recognised — inferring generic 'sphere'")
            return [ParsedEntity(name="sphere", object_type="sphere", is_actor=False)]
        # Check if there's an action verb implying a humanoid actor
        for pattern, kw in self._act_patterns:
            if pattern.search(text):
                act_def = ACTIONS[self._act_kw[kw]]
                if act_def.category not in (ActionCategory.PHYSICS,):
                    log.info("[M1-Rules] action '%s' implies humanoid actor", kw)
                    bp = body_params_from_prompt(text)
                    return [ParsedEntity(
                        name="humanoid", object_type="humanoid",
                        is_actor=True, body_params=bp,
                    )]
        return []

    def color_context(self, text: str) -> dict[str, tuple | None]:
        result: dict[str, tuple | None] = {}
        tokens = text.split()
        for i, token in enumerate(tokens):
            clean = re.sub(r"[^a-z]", "", token)
            if clean not in SCENE_COLORS:
                continue
            window = " ".join(tokens[i: i + 5])
            for pattern, kw in self._obj_patterns:
                if pattern.search(window):
                    result[kw] = SCENE_COLORS[clean]
                    break
        return result

    def actor_target(self, act_def, actors: list, objects: list) -> "tuple[str, str]":
        """Resolve actor and target names for an action definition."""
        if act_def.category != ActionCategory.PHYSICS:
            actor  = actors[0] if actors else ""
            target = objects[0] if objects and act_def.requires_target else ""
        else:
            actor  = objects[0] if objects else ""
            target = objects[1] if len(objects) > 1 and act_def.requires_target else ""
        return actor, target

    def extract_actions(self, text: str, entities: list[ParsedEntity]) -> list[ParsedAction]:
        actors  = [e.name for e in entities if e.is_actor]
        objects = [e.name for e in entities if not e.is_actor]
        seen: set[str] = set()
        actions: list[ParsedAction] = []
        for pattern, kw in self._act_patterns:
            m = pattern.search(text)
            if not m:
                continue
            act_type = self._act_kw[kw]
            if act_type in seen:
                continue
            seen.add(act_type)
            act_def = ACTIONS[act_type]
            actor, target = self.actor_target(act_def, actors, objects)
            # Capture a clause window around the matched keyword to preserve
            # directional/modifier words (e.g. "walks backward", "turns left")
            clause_start = max(0, m.start() - 30)
            clause_end   = min(len(text), m.end() + 30)
            raw_text = text[clause_start:clause_end].strip(" ,;.")
            actions.append(ParsedAction(
                action_type=act_type, actor=actor, target=target,
                raw_text=raw_text,
            ))
        return actions

    def extract_spatial(self, text: str) -> list[dict]:
        return [{"relation": _SPATIAL[p], "phrase": p} for p in self._spatial_phrases if p in text]

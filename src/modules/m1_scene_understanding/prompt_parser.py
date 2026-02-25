from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.shared.vocabulary import ACTIONS, OBJECTS, ActionCategory, ObjectCategory


@dataclass(slots=True)
class ParsedEntity:
    name: str
    object_type: str
    is_actor: bool = False
    color: Optional[Tuple[float, float, float, float]] = None


@dataclass(slots=True)
class ParsedAction:
    action_type: str
    actor: str = ""
    target: str = ""


@dataclass(slots=True)
class ParsedScene:
    """M1 -> M2 contract. Both PromptParser and StoryAgent must return this type."""
    prompt: str = ""
    entities: List[ParsedEntity] = field(default_factory=list)
    actions: List[ParsedAction] = field(default_factory=list)
    spatial_relations: List[Dict] = field(default_factory=list)
    duration: float = 5.0
    style_prompt: str = "realistic, cinematic lighting, high quality"


_COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (1.0, 0.1, 0.1, 1.0), "green":  (0.1, 0.8, 0.1, 1.0),
    "blue":   (0.1, 0.1, 1.0, 1.0), "yellow": (1.0, 0.9, 0.0, 1.0),
    "orange": (1.0, 0.5, 0.0, 1.0), "purple": (0.5, 0.0, 0.8, 1.0),
    "white":  (0.95, 0.95, 0.95, 1.0), "black": (0.1, 0.1, 0.1, 1.0),
    "gray":   (0.5, 0.5, 0.5, 1.0), "grey":   (0.5, 0.5, 0.5, 1.0),
    "brown":  (0.5, 0.3, 0.1, 1.0), "pink":   (1.0, 0.5, 0.7, 1.0),
    "cyan":   (0.0, 0.8, 0.8, 1.0),
}

_SPATIAL: Dict[str, str] = {
    "on top of": "ON", "on": "ON", "above": "ABOVE", "over": "ABOVE",
    "beneath": "UNDER", "under": "UNDER", "below": "UNDER",
    "in front of": "IN_FRONT_OF", "behind": "BEHIND",
    "beside": "BESIDE", "next to": "BESIDE", "near": "NEAR",
    "left of": "LEFT_OF", "right of": "RIGHT_OF",
    "inside": "INSIDE", "in": "INSIDE",
}

_RGBA_TO_NAME: Dict[Tuple, str] = {v: k for k, v in _COLORS.items()}


class PromptParser:
    """Rules-based, ML-free, GPU-free scene parser. Fast path for M1."""

    def __init__(self) -> None:
        self._obj_kw: Dict[str, str] = {
            kw.lower(): key for key, d in OBJECTS.items() for kw in d.keywords
        }
        self._act_kw: Dict[str, str] = {
            kw.lower(): key for key, d in ACTIONS.items() for kw in d.keywords
        }
        # Pre-compile longest-first to avoid partial keyword matches
        self._obj_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\b" + re.escape(kw) + r"\b"), kw)
            for kw in sorted(self._obj_kw, key=len, reverse=True)
        ]
        self._act_patterns: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\b" + re.escape(kw) + r"\b"), kw)
            for kw in sorted(self._act_kw, key=len, reverse=True)
        ]
        self._spatial_phrases = sorted(_SPATIAL, key=len, reverse=True)

    def setup(self) -> None: ...  # uniform interface with StoryAgent

    def parse(self, prompt: str) -> ParsedScene:
        text = prompt.lower()
        entities = self._extract_entities(text)
        return ParsedScene(
            prompt=prompt,
            entities=entities,
            actions=self._extract_actions(text, entities),
            spatial_relations=self._extract_spatial(text),
        )

    def _extract_entities(self, text: str) -> List[ParsedEntity]:
        color_ctx = self._color_context(text)
        seen: set[str] = set()
        entities: List[ParsedEntity] = []
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
            prefix = _RGBA_TO_NAME.get(color, "")
            name = f"{prefix}_{obj_type}".strip("_") if prefix else obj_type
            entities.append(ParsedEntity(name=name, object_type=obj_type, is_actor=is_actor, color=color))
        return entities

    def _color_context(self, text: str) -> Dict[str, Optional[Tuple]]:
        result: Dict[str, Optional[Tuple]] = {}
        tokens = text.split()
        for i, token in enumerate(tokens):
            clean = re.sub(r"[^a-z]", "", token)
            if clean not in _COLORS:
                continue
            window = " ".join(tokens[i: i + 5])
            for pattern, kw in self._obj_patterns:
                if pattern.search(window):
                    result[kw] = _COLORS[clean]
                    break
        return result

    def _extract_actions(self, text: str, entities: List[ParsedEntity]) -> List[ParsedAction]:
        actors  = [e.name for e in entities if e.is_actor]
        objects = [e.name for e in entities if not e.is_actor]
        seen: set[str] = set()
        actions: List[ParsedAction] = []
        for pattern, kw in self._act_patterns:
            if not pattern.search(text):
                continue
            act_type = self._act_kw[kw]
            if act_type in seen:
                continue
            seen.add(act_type)
            act_def = ACTIONS[act_type]
            if act_def.category != ActionCategory.PHYSICS:
                actor  = actors[0]  if actors  else ""
                target = objects[0] if objects and act_def.requires_target else ""
            else:
                actor  = objects[0] if objects else ""
                target = objects[1] if len(objects) > 1 and act_def.requires_target else ""
            actions.append(ParsedAction(action_type=act_type, actor=actor, target=target))
        return actions

    def _extract_spatial(self, text: str) -> List[Dict]:
        return [{"relation": _SPATIAL[p], "phrase": p} for p in self._spatial_phrases if p in text]

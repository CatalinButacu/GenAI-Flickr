"""
#WHERE
    Used by extractor.py, t5_parser.py, reasoner.py, builder.py,
    orchestrator.py, and templates.py within M1.

#WHAT
    Data models for the M1 extraction stage.  These represent the raw
    entities, actions, and relations parsed from a user prompt before
    any reasoning or scene building occurs.

#INPUT
    Raw text prompt fields (name, verb, spans, attributes, etc.).

#OUTPUT
    Typed dataclass instances: ExtractionResult containing lists of
    ExtractedEntity, ExtractedAction, and ExtractedRelation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class EntityType(Enum):
    PERSON = auto()
    ANIMAL = auto()
    VEHICLE = auto()
    OBJECT = auto()
    ENVIRONMENT = auto()


@dataclass(slots=True)
class ExtractedAttribute:
    key: str
    value: Any = None
    confidence: float = 1.0
    source: str = "extraction"


@dataclass
class ExtractedEntity:
    id: str
    name: str = ""
    entity_type: EntityType = EntityType.OBJECT
    raw_span: str = ""
    count: int = 1
    attributes: List[ExtractedAttribute] = field(default_factory=list)
    dimensions: Optional[Dict[str, float]] = None
    mass: Optional[float] = None
    material: Optional[str] = None
    mesh_prompt: Optional[str] = None
    is_static: bool = False
    parts: List[str] = field(default_factory=list)

    @property
    def color(self) -> Optional[List[float]]:
        return self.get_attr("color_vec")

    def get_attr(self, key: str) -> Optional[Any]:
        return next((a.value for a in self.attributes if a.key == key), None)

    def set_attr(self, key: str, value: Any,
                 confidence: float = 1.0, source: str = "inference") -> None:
        for attr in self.attributes:
            if attr.key == key:
                attr.value, attr.confidence, attr.source = value, confidence, source
                return
        self.attributes.append(ExtractedAttribute(key, value, confidence, source))


@dataclass(slots=True)
class ExtractedAction:
    verb: str
    raw_span: str = ""
    actor_id: Optional[str] = None
    target_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedRelation:
    source_id: str
    relation: str
    target_id: str
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    raw_prompt: str = ""
    entities: List[ExtractedEntity] = field(default_factory=list)
    actions: List[ExtractedAction] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    inferred_activity: Optional[str] = None
    inferred_setting: Optional[str] = None
    style_prompt: str = "realistic, cinematic lighting, high quality"
    duration: float = 5.0

    @property
    def entity_names(self) -> List[str]:
        return [e.name for e in self.entities]

    @property
    def has_entities(self) -> bool:
        return bool(self.entities)

    @property
    def static_entities(self) -> List[ExtractedEntity]:
        return [e for e in self.entities if e.is_static]

    @property
    def dynamic_entities(self) -> List[ExtractedEntity]:
        return [e for e in self.entities if not e.is_static]

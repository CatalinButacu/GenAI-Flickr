"""
#WHERE
    Used by reasoner.py and templates.py within M1.

#WHAT
    Data models for the M1 reasoning stage.  Templates describe
    activity patterns (e.g. "football kick") with their expected
    implicit entities, spatial hints, and trigger words.

#INPUT
    Activity template definitions with trigger verbs/nouns, implicit
    entities (e.g. a goal net for "football kick"), and spatial hints.

#OUTPUT
    Typed dataclass instances: ActivityTemplate, ImplicitEntity, SpatialHint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .extraction_models import EntityType


@dataclass(slots=True)
class ImplicitEntity:
    name: str
    entity_type: EntityType
    role: str
    required: bool = True
    default_dimensions: Optional[Dict[str, float]] = None
    default_mass: Optional[float] = None
    mesh_prompt: str = ""


@dataclass(slots=True)
class SpatialHint:
    source_role: str
    target_role: str
    relation: str
    distance_m: float = 1.0


@dataclass
class ActivityTemplate:
    name: str
    trigger_verbs: Set[str]
    trigger_nouns: Set[str] = field(default_factory=set)
    expected_person_count: Optional[int] = None
    implicit_entities: List[ImplicitEntity] = field(default_factory=list)
    spatial_hints: List[SpatialHint] = field(default_factory=list)
    default_setting: str = ""

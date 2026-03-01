"""
#WHERE
    Called by orchestrator.py (StoryAgent) between extraction and building.

#WHAT
    Enriches ExtractionResult with implicit entities from activity templates
    and knowledge base lookups (spatial hints, default attributes).

#INPUT
    ExtractionResult from Extractor, KnowledgeRetriever instance.

#OUTPUT
    Enriched ExtractionResult with additional entities and relations.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Optional

from .models import (
    ActivityTemplate, EntityType, ExtractedAttribute, ExtractedEntity,
    ExtractedRelation, ExtractionResult,
)
from .retriever import KnowledgeRetriever
from .templates import TEMPLATES

log = logging.getLogger(__name__)

# Physical defaults per entity category — used when extraction has no dimensions/mass
HUMAN_DEFAULTS = {
    "adult_male":   {"height": 1.78, "mass": 80.0},
    "adult_female": {"height": 1.65, "mass": 65.0},
    "generic":      {"height": 1.72, "mass": 72.0},
}
ANIMAL_DEFAULTS = {
    "dog":  {"height": 0.60, "mass": 25.0, "length": 0.80},
    "cat":  {"height": 0.30, "mass":  5.0, "length": 0.50},
    "horse":{"height": 1.60, "mass":500.0, "length": 2.40},
    "generic":{"height":0.50,"mass": 20.0, "length": 0.60},
}
VEHICLE_DEFAULTS = {
    "car":       {"height": 1.45, "mass": 1500.0, "length": 4.5, "width": 1.8},
    "truck":     {"height": 2.50, "mass": 8000.0, "length": 7.0, "width": 2.5},
    "motorcycle":{"height": 1.10, "mass":  200.0, "length": 2.1, "width": 0.8},
    "bicycle":   {"height": 1.00, "mass":   12.0, "length": 1.8, "width": 0.5},
    "generic":   {"height": 1.50, "mass": 1200.0, "length": 4.0, "width": 1.8},
}
MATERIAL_DENSITY = {
    "wood": 600, "metal": 7800, "plastic": 950, "glass": 2500,
    "stone": 2600, "rubber": 1100, "fabric": 300, "concrete": 2400,
}

_FEMALE_CUES = {"girlfriend", "wife", "mother", "sister", "daughter", "woman", "girl", "she", "her"}
_MALE_CUES   = {"boyfriend", "husband", "father", "brother", "son", "man", "boy", "he", "him"}
_CAR_BRANDS  = {"bmw", "mercedes", "ferrari", "porsche", "tesla", "audi", "toyota", "honda"}


class Reasoner:
    """Enriches ExtractionResult with commonsense defaults, KB data, and template inference."""

    def __init__(self, retriever: Optional[KnowledgeRetriever] = None) -> None:
        self._retriever = retriever

    @property
    def has_retriever(self) -> bool:
        return self._retriever is not None and self._retriever.is_ready

    def reason(self, extraction: ExtractionResult) -> ExtractionResult:
        entities  = list(extraction.entities)
        actions   = list(extraction.actions)
        relations = list(extraction.relations)

        template = self._match_template(extraction)
        if template:
            entities = self._apply_template(template, entities)
            log.info("Template matched: %s", template.name)

        if self.has_retriever:
            entities = self._enrich_from_kb(entities)

        entities = self._apply_category_defaults(entities)
        entities = self._estimate_mass_from_volume(entities)

        return ExtractionResult(
            raw_prompt=extraction.raw_prompt,
            entities=entities, actions=actions, relations=relations,
            inferred_activity=template.name if template else extraction.inferred_activity,
            inferred_setting=extraction.inferred_setting or (template.default_setting if template else None),
            style_prompt=extraction.style_prompt,
            duration=extraction.duration,
        )

    # -- template matching --

    def _match_template(self, extraction: ExtractionResult) -> Optional[ActivityTemplate]:
        verbs = {a.verb.lower() for a in extraction.actions}
        best, best_score = None, 0
        for t in TEMPLATES:
            score = len(verbs & t.trigger_verbs)
            if score > best_score:
                best, best_score = t, score
        return best if best_score > 0 else None

    def _apply_template(self, template: ActivityTemplate, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        existing = {e.name.lower() for e in entities}
        for impl in template.implicit_entities:
            if impl.name.lower() not in existing:
                entities.append(ExtractedEntity(
                    id=f"template_{template.name}_{impl.role}",
                    entity_type=impl.entity_type, name=impl.name, count=1,
                    attributes=[ExtractedAttribute("role", impl.role, 0.8, "template")],
                    dimensions=impl.default_dimensions,
                    mass=impl.default_mass,
                    mesh_prompt=impl.mesh_prompt,
                ))
        return entities

    # -- KB enrichment --

    @staticmethod
    def _kb_name_matches(entity_name: str, kb) -> bool:
        name_lower = entity_name.lower()
        if name_lower in kb.name.lower():
            return True
        return any(name_lower in alias.lower() for alias in kb.aliases)

    @staticmethod
    def _merge_kb(entity, kb):
        return replace(
            entity,
            dimensions=entity.dimensions or kb.dimensions,
            mass=entity.mass if entity.mass is not None else kb.mass,
            material=entity.material or (kb.material if kb.material != "unknown" else None),
            mesh_prompt=entity.mesh_prompt or kb.mesh_prompt,
            parts=entity.parts or kb.parts,
        )

    def _enrich_from_kb(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        result = []
        for entity in entities:
            hits = self._retriever.retrieve(entity.name, top_k=1)
            if hits and self._kb_name_matches(entity.name, hits[0]):
                entity = self._merge_kb(entity, hits[0])
            result.append(entity)
        return result

    # -- category defaults --

    def _apply_category_defaults(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        dispatch = {
            EntityType.PERSON:  self._person_defaults,
            EntityType.ANIMAL:  self._animal_defaults,
            EntityType.VEHICLE: self._vehicle_defaults,
        }
        return [dispatch.get(e.entity_type, lambda x: x)(e) for e in entities]

    def _person_defaults(self, e: ExtractedEntity) -> ExtractedEntity:
        gender  = _infer_gender(e)
        key     = f"adult_{gender}" if gender else "generic"
        d       = HUMAN_DEFAULTS[key]
        dims    = dict(e.dimensions or {})
        dims.setdefault("height", d["height"])
        return replace(e, mass=e.mass or d["mass"], dimensions=dims)

    def _animal_defaults(self, e: ExtractedEntity) -> ExtractedEntity:
        key  = next((k for k in ANIMAL_DEFAULTS if k != "generic" and k in e.name.lower()), "generic")
        d    = ANIMAL_DEFAULTS[key]
        dims = dict(e.dimensions or {})
        dims.setdefault("height", d["height"])
        return replace(e, mass=e.mass or d["mass"], dimensions=dims)

    def _vehicle_defaults(self, e: ExtractedEntity) -> ExtractedEntity:
        key  = next((k for k in VEHICLE_DEFAULTS if k != "generic" and k in e.name.lower()), "generic")
        d    = VEHICLE_DEFAULTS[key]
        dims = dict(e.dimensions or {})
        for k in ("height", "length", "width"):
            dims.setdefault(k, d[k])
        return replace(e, mass=e.mass or d["mass"], dimensions=dims)

    def _estimate_mass_from_volume(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        result = []
        for e in entities:
            if e.mass is None and e.dimensions:
                d = e.dimensions
                if all(k in d for k in ("height", "width", "length")):
                    density = MATERIAL_DENSITY.get((e.material or "").lower(), 800)
                    mass    = round(d["height"] * d["width"] * d["length"] * density * 0.4, 2)
                    e = replace(e, mass=mass)
            result.append(e)
        return result


def _infer_gender(entity: ExtractedEntity) -> str:
    tokens = set(entity.name.lower().split())
    if tokens & _FEMALE_CUES: return "female"
    if tokens & _MALE_CUES:   return "male"
    return ""

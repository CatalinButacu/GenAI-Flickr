"""M1 (Scene Understanding) + M2 (Scene Planning) stage."""
from __future__ import annotations

import logging
import time
from typing import Any

from src.shared.mem_profile import tracemalloc_snapshot
from src.modules.understanding.prompt_parser import PromptParser
from src.modules.understanding.t5_parser import T5SceneParser
from src.modules.understanding.retriever import KnowledgeRetriever
from src.modules.planner import ScenePlanner

log = logging.getLogger(__name__)


class UnderstandingStage:
    """M1: Parse text prompt into structured scene → M2: Plan spatial layout."""

    def __init__(self, *, use_t5: bool = True, device: str = "cuda") -> None:
        self.use_t5 = use_t5
        self.device = device
        self.parser: Any = None
        self.planner: ScenePlanner | None = None
        self.kb_retriever: KnowledgeRetriever | None = None

    def setup(self) -> None:
        self.parser = self._init_parser()
        self.planner = ScenePlanner()
        log.info("[M2] ScenePlanner ready")
        self.kb_retriever = self._init_kb()

    # ── public API ──────────────────────────────────────────────────

    def run(self, prompt: str) -> tuple:
        """Parse prompt (M1) and plan scene layout (M2).

        Returns (parsed_scene, planned_scene).
        """
        t0 = time.time()
        with tracemalloc_snapshot("M1 parse"):
            parsed = self.parser.parse(prompt)

        log.info("[M1] parsed in %.2fs — %d entities, %d actions", time.time() - t0, len(parsed.entities), len(parsed.actions))

        if self.kb_retriever is not None:
            t0 = time.time()
            self._enrich_entities(parsed)
            log.info("[M1] KB enrichment in %.2fs", time.time() - t0)

        t0 = time.time()
        with tracemalloc_snapshot("M2 plan"):
            planned = self.planner.plan(parsed)
        log.info("[M2] planned in %.2fs — %d entities positioned",
                 time.time() - t0, len(planned.entities))
        return parsed, planned

    # ── init helpers ────────────────────────────────────────────────

    def _init_parser(self):
        if self.use_t5:
            parser = T5SceneParser(device=self.device, fallback=True)
            log.info("[M1] T5SceneParser ready (ML-powered, with rules fallback)")
            return parser
        parser = PromptParser()
        log.info("[M1] PromptParser ready (rules-based)")
        return parser

    def _init_kb(self) -> KnowledgeRetriever | None:
        retriever = KnowledgeRetriever()
        if retriever.setup():
            log.info("[M1] KnowledgeRetriever ready — %d entries",
                     retriever.entry_count)
            return retriever
        log.warning("[M1] KnowledgeRetriever empty — disabled")
        return None

    def _enrich_entities(self, parsed) -> None:
        """Fill in KB-derived properties the parser missed."""
        enriched = 0
        for entity in parsed.entities:
            if not (results := self.kb_retriever.retrieve(entity.name, top_k=1)):
                continue
            kb = results[0]
            # Fill object_type from KB category if parser left it blank
            if not entity.object_type and kb.category:
                entity.object_type = kb.category
            # Fill material from KB if parser left it blank
            if not getattr(entity, 'material', None) and kb.material:
                entity.material = kb.material
            enriched += 1
            log.debug("[M1-KB] '%s' -> KB '%s' (category=%s, mass=%.1fkg)",
                      entity.name, kb.name, kb.category, kb.mass)
        if enriched:
            log.info("[M1] KB-enriched %d/%d entities",
                     enriched, len(parsed.entities))

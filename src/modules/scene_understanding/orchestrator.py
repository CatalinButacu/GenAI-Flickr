"""
#WHERE
    Called by pipeline.py for ML-powered scene understanding.

#WHAT
    StoryAgent — M1 orchestrator chaining: Extractor → Reasoner →
    SceneBuilder to produce a rich SceneDescription from text.

#INPUT
    Text prompt, optional checkpoint path and knowledge base dir.

#OUTPUT
    SceneDescription with objects, actions, camera, relations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .models import SceneDescription
from .extractor import Extractor
from .retriever import KnowledgeRetriever
from .reasoner import Reasoner
from .builder import SceneBuilder

log = logging.getLogger(__name__)


class StoryAgent:
    """M1 ML-powered orchestrator: prompt -> Extractor -> Reasoner -> SceneBuilder -> SceneDescription."""

    def __init__(self, checkpoint_path: Optional[str] = None, kb_dir: Optional[str] = None) -> None:
        self._checkpoint_path = checkpoint_path
        self._extractor = Extractor(checkpoint_path=checkpoint_path)
        self._retriever = KnowledgeRetriever(kb_dir=kb_dir)
        self._reasoner  = Reasoner(retriever=self._retriever)
        self._builder   = SceneBuilder()
        self._ready     = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def checkpoint_path(self) -> Optional[str]:
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, value: str) -> None:
        self._checkpoint_path = value
        self._extractor.checkpoint_path = value
        self._ready = False

    @property
    def extraction_mode(self) -> str:
        return "ml" if self._extractor.is_loaded else "pending"

    def setup(self) -> None:
        """Load T5 checkpoint + KB index. Raises FileNotFoundError if checkpoint is missing."""
        self._extractor.load()
        self._retriever.setup()
        self._ready = True
        log.info("StoryAgent ready — extraction=%s, kb_entries=%d", self.extraction_mode, self._retriever.entry_count)

    def parse(self, prompt: str) -> SceneDescription:
        if not self._ready:
            self.setup()
        extraction = self._extractor.extract(prompt)
        extraction = self._reasoner.reason(extraction)
        return self._builder.build(extraction)

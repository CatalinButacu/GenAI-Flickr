"""
#WHERE
    Used by generator.py (MotionGenerator._build_retriever) as the
    preferred retrieval backend when sentence-transformers is available.

#WHAT
    SBERT-powered semantic retrieval for KIT-ML motion clips.
    Replaces exact-keyword matching with cosine similarity over
    all-MiniLM-L6-v2 embeddings (~80 MB).

#INPUT
    Text query (e.g. "stroll", "march"), max frame count.

#OUTPUT
    MotionClip with the semantically closest KIT-ML sample.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .constants import DEFAULT_DATA_DIR
from .models import MotionClip
from .keyword_retriever import MotionRetriever

log = logging.getLogger(__name__)

_SBERT_MODEL = "all-MiniLM-L6-v2"


class SemanticRetriever(MotionRetriever):
    """KIT-ML retriever with SBERT cosine-similarity matching.

    Inherits dataset loading from :class:`MotionRetriever` and adds an
    embedding index over the text annotations for semantic look-up.
    """

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, device: str = "cpu"):
        super().__init__(data_dir)
        self._sbert = None
        self._embeddings = None
        self._device = device
        self._load_sbert()

    # ------------------------------------------------------------------
    # Embedding index
    # ------------------------------------------------------------------

    def _load_sbert(self) -> None:
        if not self._samples:
            log.warning("SemanticRetriever: no samples loaded — SBERT skipped")
            return
        from sentence_transformers import SentenceTransformer

        self._sbert = SentenceTransformer(_SBERT_MODEL, device=self._device)
        texts = [s.text for s in self._samples]
        self._embeddings = self._sbert.encode(
            texts, convert_to_tensor=True, show_progress_bar=False,
        )
        log.info(
            "SemanticRetriever: indexed %d samples with %s",
            len(texts), _SBERT_MODEL,
        )

    # ------------------------------------------------------------------
    # Public API (overrides MotionRetriever.retrieve)
    # ------------------------------------------------------------------

    def retrieve(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        """Retrieve the best-matching KIT-ML clip via cosine similarity.

        Falls back to keyword matching if SBERT is unavailable.
        """
        if self._sbert is None or self._embeddings is None:
            return super().retrieve(text, max_frames)

        from sentence_transformers import util

        query_emb = self._sbert.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self._embeddings)[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        sample = self._samples[best_idx]
        raw = self._load_raw_joints(sample.sample_id, max_frames)

        log.debug(
            "SemanticRetriever: '%s' → '%s' (score=%.3f)",
            text, sample.text[:60], best_score,
        )
        return MotionClip(
            action=text,
            frames=sample.motion[:max_frames],
            source="semantic_retrieved",
            raw_joints=raw,
        )

"""
#WHERE
    Used by reasoner.py and orchestrator.py (StoryAgent) for scene enrichment.

#WHAT
    SBERT + FAISS knowledge retrieval from Visual Genome knowledge base.
    Retrieves spatial relations, object affordances, and scene context.

#INPUT
    Text query, top-k count.

#OUTPUT
    List of dicts with retrieved knowledge triples and similarity scores.
"""

from __future__ import annotations

import json
import logging
import os
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

log = logging.getLogger(__name__)

_DEFAULT_KB_DIR    = Path("data/knowledge_base")
_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"


class KBEntry:
    """Single knowledge-base record with lazy property access."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    @property
    def name(self) -> str:
        return self._data.get("canonical_name", "")

    @name.setter
    def name(self, value: str) -> None:
        self._data["canonical_name"] = value

    @property
    def aliases(self) -> List[str]:
        return self._data.get("aliases", [])

    @property
    def category(self) -> str:
        return self._data.get("category", "unknown")

    @property
    def dimensions(self) -> Dict[str, float]:
        return self._data.get("typical_dimensions_m", {})

    @property
    def mass(self) -> float:
        return self._data.get("typical_mass_kg", 1.0)

    @property
    def material(self) -> str:
        return self._data.get("material", "unknown")

    @property
    def parts(self) -> List[str]:
        return self._data.get("parts", [])

    @property
    def mesh_prompt(self) -> str:
        return self._data.get("mesh_prompt", "")

    @property
    def related_objects(self) -> List[str]:
        return self._data.get("related_objects", [])

    def __repr__(self) -> str:
        return f"KBEntry(name={self.name!r}, category={self.category!r})"


class KnowledgeRetriever:
    """
    RAG retriever over a local knowledge base.

    Encoding: SentenceTransformer embeds entry text → FAISS IndexFlatIP (cosine similarity).
    Querying: encode(query) → index.search → top-k KBEntry results.

    Single data path: FAISS vector search. The retriever is either ready or it isn't.
    """

    def __init__(self, kb_dir: Optional[str] = None) -> None:
        self._kb_dir   = Path(kb_dir) if kb_dir else _DEFAULT_KB_DIR
        self._entries: List[KBEntry] = []
        self._texts:   List[str]     = []
        self._encoder: Optional[SentenceTransformer] = None
        self._index    = None

    # -- observable state --

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._entries) > 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def kb_dir(self) -> Path:
        return self._kb_dir

    # -- setup --

    def setup(self) -> bool:
        self._load_entries()
        if not self._entries:
            log.warning("Knowledge base empty at %s", self._kb_dir / "objects")
            return False
        self._init_vector_index()
        log.info("KnowledgeRetriever ready — %d entries, FAISS index active", self.entry_count)
        return True

    # -- retrieval --

    def retrieve(self, query: str, top_k: int = 1) -> List[KBEntry]:
        assert self._index is not None, (
            "KnowledgeRetriever not ready — call setup() first."
        )
        return self._vector_search(query, top_k)

    # -- private --

    def _load_entries(self) -> None:
        objects_dir = self._kb_dir / "objects"
        if not objects_dir.exists():
            return
        for path in sorted(objects_dir.glob("*.json")):
            data  = json.loads(path.read_text(encoding="utf-8"))
            items = list(data.values()) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for item in items:
                entry = KBEntry(item)
                self._entries.append(entry)
                self._texts.append(
                    f"{entry.name} {' '.join(entry.aliases)} {entry.category} {entry.mesh_prompt}"
                )
            log.debug("Loaded %d entries from %s", len(items), path.name)

    def _init_vector_index(self) -> None:
        self._encoder = SentenceTransformer(_EMBEDDING_MODEL)

        index_path = self._kb_dir / "embeddings" / "object_index.faiss"
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
            log.info("Loaded FAISS index (%d entries) from %s", self.entry_count, index_path)
        else:
            log.info("Building FAISS index for %d entries…", self.entry_count)
            vecs = self._encoder.encode(
                self._texts, show_progress_bar=False, convert_to_numpy=True
            ).astype(np.float32)
            faiss.normalize_L2(vecs)
            self._index = faiss.IndexFlatIP(vecs.shape[1])
            self._index.add(vecs)

    def _vector_search(self, query: str, top_k: int) -> List[KBEntry]:
        q = self._encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        _, idxs = self._index.search(q, min(top_k, self.entry_count))
        return [self._entries[i] for i in idxs[0] if 0 <= i < self.entry_count]

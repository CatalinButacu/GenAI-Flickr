
from __future__ import annotations

import json
import logging
import os
import numpy as np
import faiss
from dataclasses import dataclass, field
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Any

# Suppress TensorFlow noise — deferred to first use instead of import time
def _suppress_tf_warnings() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

log = logging.getLogger(__name__)

_DEFAULT_KB_DIR    = Path("data/knowledge_base")
_EMBEDDING_MODEL   = "all-MiniLM-L6-v2"


@dataclass(slots=True)
class KBEntry:
    """Single knowledge-base record."""
    name: str                          = ""
    aliases: list[str]                 = field(default_factory=list)
    category: str                      = "unknown"
    dimensions: dict[str, float]       = field(default_factory=dict)
    mass: float                        = 1.0
    material: str                      = "unknown"
    parts: list[str]                   = field(default_factory=list)
    mesh_prompt: str                   = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KBEntry:
        """Construct from a raw JSON dict (keys differ from field names)."""
        return cls(
            name=data.get("canonical_name", ""),
            aliases=data.get("aliases", []),
            category=data.get("category", "unknown"),
            dimensions=data.get("typical_dimensions_m", {}),
            mass=data.get("typical_mass_kg", 1.0),
            material=data.get("material", "unknown"),
            parts=data.get("parts", []),
            mesh_prompt=data.get("mesh_prompt", ""),
        )


class KnowledgeRetriever:
    """
    RAG retriever over a local knowledge base.

    Encoding: SentenceTransformer embeds entry text → FAISS IndexFlatIP (cosine similarity).
    Querying: encode(query) → index.search → top-k KBEntry results.

    Single data path: FAISS vector search. The retriever is either ready or it isn't.
    """

    def __init__(self, kb_dir: str | None = None) -> None:
        self._kb_dir   = Path(kb_dir) if kb_dir else _DEFAULT_KB_DIR
        self._entries: list[KBEntry] = []
        self._texts:   list[str]     = []
        self._encoder: SentenceTransformer | None = None
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
        self.load_entries()
        if not self._entries:
            log.warning("Knowledge base empty at %s", self._kb_dir / "objects")
            return False
        self.init_vector_index()
        log.info("KnowledgeRetriever ready — %d entries, FAISS index active", self.entry_count)
        return True

    # -- retrieval --

    def retrieve(self, query: str, top_k: int = 1) -> list[KBEntry]:
        if self._index is None:
            raise RuntimeError("KnowledgeRetriever not ready — call setup() first.")
        return self.vector_search(query, top_k)

    # -- private --

    def load_entries(self) -> None:
        objects_dir = self._kb_dir / "objects"
        if not objects_dir.exists():
            return
        for path in sorted(objects_dir.glob("*.json")):
            data  = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                items = list(data.values())
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for item in items:
                entry = KBEntry.from_dict(item)
                self._entries.append(entry)
                self._texts.append(
                    f"{entry.name} {' '.join(entry.aliases)} {entry.category} {entry.mesh_prompt}"
                )
            log.debug("Loaded %d entries from %s", len(items), path.name)

    def init_vector_index(self) -> None:
        _suppress_tf_warnings()
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
            # Persist so future runs skip re-encoding
            self.save_index()

    def save_index(self) -> None:
        """Persist the FAISS index to disk."""
        if self._index is None:
            return
        index_path = self._kb_dir / "embeddings" / "object_index.faiss"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        log.info("FAISS index saved → %s", index_path)

    def vector_search(self, query: str, top_k: int) -> list[KBEntry]:
        q = self._encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q)
        _, idxs = self._index.search(q, min(top_k, self.entry_count))
        return [self._entries[i] for i in idxs[0] if 0 <= i < self.entry_count]

"""
VG Knowledge-Base Index Builder (Single Responsibility: Index Construction)
===========================================================================
Reads Visual Genome objects.json and builds a FAISS vector index + vocabulary
JSON that the KnowledgeRetriever can load for semantic object lookup.

Output:
    data/knowledge_base/embeddings/vg_object_index.faiss
    data/knowledge_base/vg_vocab.json

Usage:
    python scripts/build_vg_kb_index.py
    python scripts/build_vg_kb_index.py --max-objects 10000
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VG object frequency counter  (SRP: "count VG object frequency")
# ---------------------------------------------------------------------------

class VGObjectCounter:
    """Reads objects.json and returns the most common object names."""

    def __init__(self, objects_path: Path) -> None:
        self._path = objects_path

    def get_top_names(self, top_k: int) -> List[Tuple[str, int]]:
        log.info("Loading objects.json (~349 MB) …")
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)

        counter: collections.Counter = collections.Counter()
        for image in data:
            for obj in image.get("objects", []):
                for name in obj.get("names", []):
                    cleaned = name.strip().lower()
                    if cleaned and len(cleaned) >= 2:
                        counter[cleaned] += 1

        log.info("Found %d unique object names. Picking top %d …", len(counter), top_k)
        return counter.most_common(top_k)


# ---------------------------------------------------------------------------
# Object alias enricher  (SRP: "load & apply VG alias mappings")
# ---------------------------------------------------------------------------

class VGAliasLoader:
    """Reads object_alias.txt to get synonym groups."""

    def __init__(self, alias_path: Path) -> None:
        self._path = alias_path
        self._aliases: dict[str, List[str]] = {}

    def load(self) -> None:
        if not self._path.exists():
            log.warning("object_alias.txt not found at %s — skipping aliases", self._path)
            return
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                parts = [p.strip().lower() for p in line.strip().split(",") if p.strip()]
                if len(parts) >= 2:
                    canonical = parts[0]
                    self._aliases[canonical] = parts[1:]

    def get_aliases(self, name: str) -> List[str]:
        return self._aliases.get(name.lower(), [])


# ---------------------------------------------------------------------------
# Embedding encoder  (SRP: "encode text to vectors")
# ---------------------------------------------------------------------------

class TextEncoder:
    """Wraps SentenceTransformer to produce L2-normalised embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)
        log.info("Loaded embedding model: %s", self._model_name)

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        assert self._model is not None, "Call load() first"
        log.info("Encoding %d texts (batch_size=%d) …", len(texts), batch_size)
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise for cosine via IndexFlatIP
        )
        return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# FAISS index builder  (SRP: "build and save FAISS index")
# ---------------------------------------------------------------------------

class FAISSIndexBuilder:
    """Builds and persists a FAISS IndexFlatIP (inner product = cosine on L2-normed vecs)."""

    def build_and_save(self, embeddings: np.ndarray, output_path: Path) -> None:
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))
        log.info("FAISS index saved → %s  (vectors=%d, dim=%d)",
                 output_path, index.ntotal, dim)


# ---------------------------------------------------------------------------
# Vocabulary JSON writer  (SRP: "persist vocab metadata")
# ---------------------------------------------------------------------------

class VocabWriter:
    """Persists the vocabulary list and alias map as JSON."""

    def write(self, vocab_entries: List[dict], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab_entries, f, ensure_ascii=False, indent=2)
        log.info("Vocabulary saved → %s  (%d entries)", output_path, len(vocab_entries))


# ---------------------------------------------------------------------------
# KB index pipeline orchestrator
# ---------------------------------------------------------------------------

class KBIndexPipeline:
    """
    Orchestrates: VG objects → frequency count → embed → FAISS index.
    SRP: pipeline coordination only.
    """

    def __init__(
        self,
        vg_dir: Path,
        output_dir: Path,
        max_objects: int = 10_000,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._vg_dir    = vg_dir
        self._out_dir   = output_dir
        self._max       = max_objects
        self._counter   = VGObjectCounter(vg_dir / "objects.json")
        self._alias_ldr = VGAliasLoader(vg_dir / "object_alias.txt")
        self._encoder   = TextEncoder(embedding_model)
        self._idx_bld   = FAISSIndexBuilder()
        self._vocab_wtr = VocabWriter()

    def run(self) -> None:
        # 1. Frequency count
        top_names = self._counter.get_top_names(self._max)

        # 2. Load aliases
        self._alias_ldr.load()

        # 3. Build embed texts = "name alias1 alias2 ..."
        names: List[str] = []
        vocab_entries: List[dict] = []
        for name, freq in top_names:
            aliases = self._alias_ldr.get_aliases(name)
            embed_text = " ".join([name] + aliases[:3])
            names.append(embed_text)
            vocab_entries.append({
                "canonical_name": name,
                "aliases": aliases,
                "frequency": freq,
                "category": "unknown",
                "typical_mass_kg": 1.0,
                "typical_dimensions_m": {},
                "material": "unknown",
                "parts": [],
                "physics": {},
                "mesh_prompt": f"a realistic {name}",
                "common_contexts": [],
                "related_objects": [],
            })

        # 4. Encode
        self._encoder.load()
        embeddings = self._encoder.encode(names)

        # 5. Save FAISS index
        index_path = self._out_dir / "embeddings" / "vg_object_index.faiss"
        self._idx_bld.build_and_save(embeddings, index_path)

        # 6. Save vocabulary JSON
        vocab_path = self._out_dir / "vg_vocab.json"
        self._vocab_wtr.write(vocab_entries, vocab_path)

        # 7. Also write a minimal KB objects file the retriever can load
        self._write_kb_objects_file(vocab_entries)

        log.info("KB index pipeline complete.")

    def _write_kb_objects_file(self, entries: List[dict]) -> None:
        """Write entries as a KB objects JSON file compatible with KnowledgeRetriever."""
        kb_objects_dir = self._out_dir / "objects"
        kb_objects_dir.mkdir(parents=True, exist_ok=True)
        # Split into chunks of 1000 to avoid huge single files
        chunk_size = 1000
        for i in range(0, len(entries), chunk_size):
            chunk = entries[i:i + chunk_size]
            chunk_path = kb_objects_dir / f"vg_objects_{i // chunk_size:04d}.json"
            with open(chunk_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False)
        log.info("Wrote %d KB object files → %s", len(entries) // chunk_size + 1, kb_objects_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FAISS KB index from VG objects.json")
    p.add_argument("--vg-dir",      default="data/M1_VisualGenome",
                   help="Directory containing VG JSON files")
    p.add_argument("--output",      default="data/knowledge_base",
                   help="Output directory for FAISS index and vocab (default: data/knowledge_base)")
    p.add_argument("--max-objects", type=int, default=10_000,
                   help="Max object types to index (default: 10 000)")
    p.add_argument("--embedding-model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="HuggingFace SentenceTransformer model name")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipeline = KBIndexPipeline(
        vg_dir=Path(args.vg_dir),
        output_dir=Path(args.output),
        max_objects=args.max_objects,
        embedding_model=args.embedding_model,
    )
    pipeline.run()

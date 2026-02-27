"""
#WHERE
    Used by generator.py (as the default retriever) and semantic_retriever.py
    (which extends this class).

#WHAT
    Keyword-based motion retrieval from the KIT-ML dataset.  Indexes samples
    by action keywords and retrieves the closest match by simple string search.

#INPUT
    text query (e.g. "a person walks forward"), max frame count.

#OUTPUT
    MotionClip with retrieved motion features and optional raw joints,
    or None if no matching sample is found.
"""

import logging
import random
from typing import Optional

import numpy as np

from .constants import DEFAULT_DATA_DIR, INDEX_ACTIONS
from .models import MotionClip

log = logging.getLogger(__name__)


class MotionRetriever:
    """KIT-ML dataset retrieval backend (keyword matching)."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self._index: dict[str, list] = {}
        self._samples: list = []
        self._data_dir = data_dir
        self._load(data_dir)

    @property
    def action_index(self) -> dict[str, list]:
        return self._index

    @property
    def samples(self) -> list:
        return self._samples

    def _load(self, data_dir: str) -> None:
        try:
            from src.data import KITMLLoader
            dataset = KITMLLoader(data_dir).load_dataset("train", normalize=False)
            self._samples = dataset.samples
            for s in self._samples:
                tl = s.text.lower()
                for act in INDEX_ACTIONS:
                    if act in tl:
                        self._index.setdefault(act, []).append(s)
            log.info("MotionRetriever: %d samples indexed", len(self._samples))
        except Exception as e:
            log.warning("Cannot load KIT-ML: %s", e)

    def retrieve(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        tl = text.lower()
        sample = None
        for act, samples in self._index.items():
            if act in tl:
                sample = random.choice(samples)
                break
        if sample is None and self._samples:
            sample = random.choice(self._samples)
        if sample is None:
            return None
        raw = self._load_raw_joints(sample.sample_id, max_frames)
        return MotionClip(
            action=text, frames=sample.motion[:max_frames],
            source="retrieved", raw_joints=raw,
        )

    def _load_raw_joints(self, sample_id: str, max_frames: int) -> "np.ndarray | None":
        """Load (T, 21, 3) raw joint positions from new_joints/ if available."""
        import pathlib
        p = pathlib.Path(self._data_dir) / "new_joints" / f"{sample_id}.npy"
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 3 and arr.shape[1] == 21:
                return arr[:max_frames]
        return None

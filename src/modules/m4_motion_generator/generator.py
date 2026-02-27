"""
#WHERE
    Re-export shim — keeps existing `from .generator import X` working.
    New code should import from models, keyword_retriever, ssm_model directly.

#WHAT
    Unified text-to-motion generation orchestrator.
    Falls back through: semantic/keyword retrieval → SSM → placeholder.

#INPUT
    Text prompt, desired frame count, preferred backend name.

#OUTPUT
    MotionClip with action label, (T, MOTION_DIM) features, and source tag.
"""

import logging
from typing import Optional

import numpy as np

from src.shared.mem_profile import profile_memory
from .constants import DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT, MOTION_DIM
from .models import MotionClip                    # noqa: F401  — re-export
from .keyword_retriever import MotionRetriever    # noqa: F401  — re-export
from .ssm_model import SSMMotionModel             # noqa: F401  — re-export

log = logging.getLogger(__name__)


class MotionGenerator:
    """Unified backend: semantic/keyword retrieval > SSM > placeholder."""

    def __init__(self, use_retrieval: bool = True, use_ssm: bool = True,
                 use_semantic: bool = True,
                 data_dir: str = DEFAULT_DATA_DIR,
                 checkpoint_path: str = DEFAULT_SSM_CHECKPOINT):
        self._retriever: Optional[MotionRetriever] = None
        self._ssm: Optional[SSMMotionModel] = (
            SSMMotionModel(checkpoint_path, data_dir) if use_ssm else None
        )
        if use_retrieval:
            self._retriever = self._build_retriever(data_dir, use_semantic)

    @property
    def retriever(self) -> Optional[MotionRetriever]:
        return self._retriever

    @property
    def ssm_model(self) -> Optional[SSMMotionModel]:
        return self._ssm

    @staticmethod
    def _build_retriever(data_dir: str, use_semantic: bool) -> MotionRetriever:
        """Create the best available retriever: SBERT > keyword."""
        if use_semantic:
            try:
                from .semantic_retriever import SemanticRetriever
                return SemanticRetriever(data_dir)
            except Exception as exc:
                log.warning("SemanticRetriever unavailable (%s) — keyword fallback", exc)
        return MotionRetriever(data_dir)

    @profile_memory
    def generate(self, text: str, num_frames: int = 100,
                 prefer: str = "retrieval") -> MotionClip:
        if prefer == "retrieval" and self._retriever:
            if clip := self._retriever.retrieve(text, num_frames):
                return clip
        if prefer in ("ssm", "retrieval") and self._ssm and self._ssm.model:
            if clip := self._ssm.generate(text, num_frames):
                return clip
        return self._placeholder(text, num_frames)

    @staticmethod
    def _placeholder(text: str, num_frames: int) -> MotionClip:
        t = np.linspace(0, 4 * np.pi, num_frames)
        motion = np.zeros((num_frames, MOTION_DIM))
        motion[:, 0] = 0.01 * np.sin(t)
        motion[:, 2] = 0.1 + 0.05 * np.sin(t * 0.5)
        return MotionClip(action=text, frames=motion, source="placeholder")


def create_motion_generator(**kwargs) -> MotionGenerator:
    return MotionGenerator(**kwargs)

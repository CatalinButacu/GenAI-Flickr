"""
#WHERE
    Used by generator.py, keyword_retriever.py, semantic_retriever.py,
    ssm_model.py, ssm_generator.py, pipeline.py, and benchmarks.

#WHAT
    Core data model for a motion clip — a time-series of HumanML3D feature
    vectors (T, 251) with metadata (action label, fps, source, raw joints).

#INPUT
    action string, numpy frames array, optional raw joint positions.

#OUTPUT
    MotionClip dataclass instance.
"""

from dataclasses import dataclass

import numpy as np

from .constants import MOTION_FPS


@dataclass
class MotionClip:
    action: str
    frames: np.ndarray                      # (T, MOTION_DIM) HumanML3D features
    fps: int    = MOTION_FPS
    source: str = "generated"
    raw_joints: "np.ndarray | None" = None  # (T, 21, 3) mm Y-up — for physics

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    @property
    def num_frames(self) -> int:
        return len(self.frames)

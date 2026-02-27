"""
#WHERE
    Imported by pipeline.py, benchmarks, demo scripts, training scripts.

#WHAT
    Motion Generator Module (Module 4) — generates humanoid motion from text.
    Backends: semantic/keyword retrieval, trained SSM, placeholder fallback.

#INPUT
    Text description of a human action (e.g. "a person walks forward").

#OUTPUT
    MotionClip with (T, 251) HumanML3D features, fps, source tag.
"""

from .models import MotionClip
from .keyword_retriever import MotionRetriever
from .ssm_model import SSMMotionModel
from .generator import MotionGenerator, create_motion_generator
from .semantic_retriever import SemanticRetriever
from .constants import MOTION_DIM, MOTION_FPS, DEFAULT_DATA_DIR

__all__ = [
    "MotionGenerator",
    "MotionClip",
    "MotionRetriever",
    "SemanticRetriever",
    "SSMMotionModel",
    "create_motion_generator",
    "MOTION_DIM",
    "MOTION_FPS",
    "DEFAULT_DATA_DIR",
]

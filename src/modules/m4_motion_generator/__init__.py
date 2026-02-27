"""
Motion Generator Module (Module 4)
===================================
Generates humanoid motion from text descriptions.

Backends:
- "retrieval": Fetch ground truth from KIT-ML dataset (keyword or SBERT semantic)
- "ssm": Trained State Space Model
- "placeholder": Simple procedural motion (fallback)

Example:
    from src.modules.m4_motion_generator import MotionGenerator, MotionClip
    
    gen = MotionGenerator()
    clip = gen.generate("a person walks forward", prefer="retrieval")
    print(f"Generated {clip.num_frames} frames from {clip.source}")
"""

from .generator import (
    MotionGenerator,
    MotionClip,
    MotionRetriever,
    SSMMotionModel,
    create_motion_generator,
)
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

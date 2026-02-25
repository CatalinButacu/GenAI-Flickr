"""
Motion Generator Module (Module 4)
===================================
Generates humanoid motion from text descriptions.

Backends:
- "retrieval": Fetch ground truth from KIT-ML dataset (most reliable)
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

__all__ = [
    "MotionGenerator",
    "MotionClip", 
    "MotionRetriever",
    "SSMMotionModel",
    "create_motion_generator",
]

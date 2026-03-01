"""
#WHERE
    Imported by pipeline.py when use_ai_enhancement=True.

#WHAT
    AI Enhancer Module (Module 8) — SD 1.5 + ControlNet + AnimateDiff
    for photorealistic frame/video generation from physics skeletons.

#INPUT
    Physics-verified skeleton frames from M5, scene prompt.

#OUTPUT
    Photorealistic RGB frames/video.
"""

from .renderer import VideoRenderer
from .controlnet_human import ControlNetHumanRenderer, SkeletonProjector
from .animatediff_human import AnimateDiffHumanRenderer

__all__ = [
    'VideoRenderer',
    'ControlNetHumanRenderer',
    'AnimateDiffHumanRenderer',
    'SkeletonProjector',
]

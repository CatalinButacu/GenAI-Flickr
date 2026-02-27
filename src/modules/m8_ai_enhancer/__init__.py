# Video Renderer Module
"""
This module enhances physics simulation renders using AI.

- **AnimateDiffHumanRenderer** — preferred: temporally-consistent video
  via SD 1.5 + ControlNet OpenPose + AnimateDiff motion adapter.
  (Guo et al. 2023, arXiv:2307.04725)

- **ControlNetHumanRenderer** — fallback: per-frame SD 1.5 + ControlNet.
  Good single frames but flickers between frames.

- **VideoRenderer** — legacy depth-conditioned ControlNet.
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

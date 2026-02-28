"""
#WHERE
    Imported by pipeline.py, demo scripts, benchmarks, test_physics_engine.py.

#WHAT
    Physics Engine Module (Module 5) — PyBullet physics simulation with
    cinematic camera, humanoid body, motion retargeting, and skeleton rendering.

#INPUT
    PlannedScene from M2, MotionClip from M4, camera config.

#OUTPUT
    List[FrameData] with per-frame RGB/depth/seg, or MP4 video.
"""

from .scene import Scene
from .camera import CameraConfig, CinematicCamera, FrameData
from .simulator import Simulator
from .humanoid import HumanoidBody, HumanoidConfig, load_humanoid
from .motion_retarget import retarget_frame, retarget_sequence, pelvis_transform
from .physics_renderer import (
    PhysicsSkeletonRenderer, physics_links_to_skeleton, BONES,
    auto_orient_skeleton,
)

__all__ = [
    'Scene', 'Simulator', 'CameraConfig', 'CinematicCamera', 'FrameData',
    'HumanoidBody', 'HumanoidConfig', 'load_humanoid',
    'retarget_frame', 'retarget_sequence', 'pelvis_transform',
    'PhysicsSkeletonRenderer', 'physics_links_to_skeleton', 'BONES',
    'auto_orient_skeleton',
]

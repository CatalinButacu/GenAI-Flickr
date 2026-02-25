# Physics Engine Module
"""
This module handles 3D physics simulation using PyBullet.
It loads 3D models, simulates physics, and renders frames.
"""

from .scene import Scene
from .simulator import Simulator, CameraConfig, CinematicCamera, FrameData

__all__ = ['Scene', 'Simulator', 'CameraConfig', 'CinematicCamera', 'FrameData']

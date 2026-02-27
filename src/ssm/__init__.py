"""
#WHERE
    Imported by src/modules/m4_motion_generator (train, ssm_generator),
    src/pipeline.py, tests/test_modules.py, benchmarks.

#WHAT
    Public interface for the State Space Models package.
    Re-exports every SSM component from its dedicated sub-module.

#INPUT
    n/a (package init)

#OUTPUT
    All SSM classes, factories, and utilities available via `from src.ssm import ...`
"""

import torch

HAS_TORCH = True

from .config import SSMConfig
from .s4 import S4Layer
from .mamba import MambaLayer
from .temporal import TemporalConsistencySSM
from .motion_ssm import MotionSSM
from .physics_ssm import PhysicsSSM
from .numpy_ssm import SimpleSSMNumpy
from .factory import create_ssm_layer, get_ssm_info

__all__ = [
    "SSMConfig",
    "S4Layer",
    "MambaLayer",
    "TemporalConsistencySSM",
    "MotionSSM",
    "PhysicsSSM",
    "SimpleSSMNumpy",
    "create_ssm_layer",
    "get_ssm_info",
    "HAS_TORCH",
]

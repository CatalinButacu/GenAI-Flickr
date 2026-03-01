"""
#WHERE
    Used by s4.py, mamba.py, motion_ssm.py, physics_ssm.py, temporal.py,
    factory.py, and src/modules/motion_generator/train.py

#WHAT
    SSM (State Space Model) configuration dataclass.
    Stores hyperparameters shared across all SSM layer variants.

#INPUT
    Model hyperparameters: d_model, d_state, d_conv, expand, dt_rank, etc.

#OUTPUT
    SSMConfig instance with auto-computed d_inner and dt_rank.
"""

import math
from dataclasses import dataclass


@dataclass
class SSMConfig:
    """
    Configuration for State Space Model layers.

    Attributes:
        d_model: Model dimension (hidden size)
        d_state: SSM state dimension (N in papers)
        d_conv: Convolution kernel size for Mamba
        expand: Expansion factor for inner dimension
        dt_rank: Rank of delta projection

    Reference: Mamba paper, Section 3.3
    https://arxiv.org/abs/2312.00752
    """
    d_model: int = 256
    d_state: int = 16       # N in S4/Mamba notation
    d_conv: int = 4         # Local convolution width
    expand: int = 2         # E in Mamba, inner dim = E * d_model
    dt_rank: str = "auto"   # Rank of Δ projection
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"

    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)

"""
#WHERE
    Used by src/modules/m4_motion_generator/train.py (TextToMotionSSM),
    src/modules/m4_motion_generator/ssm_generator.py, and factory.py.

#WHAT
    SSM-based motion generation module inspired by Motion Mamba (ECCV 2024).
    Stacks multiple Mamba layers with residual connections and layer norms
    to model temporal evolution of human motion sequences.

#INPUT
    Motion features tensor of shape (batch, frames, d_model).

#OUTPUT
    Processed motion tensor of shape (batch, frames, d_model).

Reference:
    Motion Mamba: "Efficient and Long Sequence Motion Generation"
    Zhang et al., ECCV 2024  —  https://arxiv.org/abs/2403.07487
"""

import torch
import torch.nn as nn

from .config import SSMConfig
from .mamba import MambaLayer


class MotionSSM(nn.Module):
    """Multi-layer Mamba stack for motion temporal modelling."""

    def __init__(self, d_model: int = 256, d_state: int = 32, n_layers: int = 4):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList([
            MambaLayer(SSMConfig(d_model=d_model, d_state=d_state))
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, frames, d_model) → (batch, frames, d_model)."""
        for layer, norm in zip(self.layers, self.norms):
            x = layer(norm(x)) + x      # pre-norm residual
        return self.output_proj(x)

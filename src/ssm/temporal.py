"""
#WHERE
    Used by factory.py.  Designed for src/modules/m8_ai_enhancer (frame consistency).

#WHAT
    SSM layer for ensuring temporal consistency in video/motion sequences.
    The hidden state carries information forward across frames, naturally
    enforcing temporal coherence (analogous to a Kalman filter).

#INPUT
    Frame features tensor of shape (batch, num_frames, d_model).

#OUTPUT
    Temporally-smoothed features of same shape (batch, num_frames, d_model).

Reference:
    VideoMamba for video understanding — https://arxiv.org/abs/2403.06977
"""

import torch
import torch.nn as nn

from .config import SSMConfig
from .mamba import MambaLayer
from .s4 import S4Layer


class TemporalConsistencySSM(nn.Module):
    """SSM wrapper for temporal consistency with residual connection."""

    def __init__(self, d_model: int = 512, d_state: int = 64, use_mamba: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_mamba = use_mamba

        if use_mamba:
            self.ssm = MambaLayer(SSMConfig(d_model=d_model, d_state=d_state))
        else:
            self.ssm = S4Layer(d_model, d_state)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """(batch, num_frames, d_model) → (batch, num_frames, d_model)."""
        residual = frames
        return self.ssm(self.norm(frames)) + residual

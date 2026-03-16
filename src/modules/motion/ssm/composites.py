from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .config import SSMConfig
from .mamba import MambaLayer
from .s4 import S4Layer
from src.shared.constants import SSM_D_MODEL, SSM_D_STATE, SSM_N_LAYERS, D_PHYSICS, GRAVITY

logger = logging.getLogger(__name__)


# ── Temporal Consistency (M8 video) ──────────────────────────────────

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


# ── Motion SSM (M4 motion generation) ───────────────────────────────

class MotionSSM(nn.Module):
    """Multi-layer Mamba stack for motion temporal modelling."""

    def __init__(self, d_model: int = SSM_D_MODEL, d_state: int = SSM_D_STATE, n_layers: int = SSM_N_LAYERS):
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


# ── Physics-Constrained SSM (novel contribution) ────────────────────

class PhysicsSSM(nn.Module):
    """
    Physics-guided SSM where hidden state evolution is influenced
    by physics simulation constraints through a learned gate:

        gate = sigmoid(W [ssm_out ; physics_embed])
        output = gate * ssm_out + (1 - gate) * constraints
    """

    def __init__(
        self,
        d_model: int = SSM_D_MODEL,
        d_state: int = SSM_D_STATE,
        d_physics: int = D_PHYSICS,
        gravity: float = GRAVITY,
    ):
        super().__init__()
        self.d_model = d_model
        self.gravity = gravity

        self.ssm = MambaLayer(SSMConfig(d_model=d_model, d_state=d_state))

        self.physics_encoder = nn.Sequential(
            nn.Linear(d_physics, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.constraint_proj = nn.Linear(d_physics, d_model)
        self.physics_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

        logger.info("PhysicsSSM initialised — novel physics-constrained architecture")

    def forward(
        self,
        motion: torch.Tensor,
        physics_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(batch, frames, d_model) → (batch, frames, d_model)."""
        ssm_out = self.ssm(self.norm(motion))

        if physics_state is not None:
            phys_embed = self.physics_encoder(physics_state)
            constraints = self.constraint_proj(physics_state)
            gate = self.physics_gate(torch.cat([ssm_out, phys_embed], dim=-1))
            output = gate * ssm_out + (1 - gate) * constraints
        else:
            output = ssm_out

        return motion + output  # residual

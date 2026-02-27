"""
#WHERE
    Used by src/modules/m4_motion_generator/ssm_generator.py and factory.py.

#WHAT
    Physics-Constrained State Space Model (PCSSM) — NOVEL CONTRIBUTION.
    Blends learned SSM dynamics with physics simulation constraints via
    a learnable gate.  The physics embedding acts as a "forcing term"
    that guides motion to be physically plausible.

#INPUT
    motion        — (batch, frames, d_model): motion feature sequence.
    physics_state — (batch, frames, d_physics): positions, velocities,
                    contact forces from the physics simulator (optional).

#OUTPUT
    Physics-constrained motion tensor of shape (batch, frames, d_model).

References:
    Physics-Informed Neural Networks — Raissi et al., 2019
    Neural ODEs — Chen et al., NeurIPS 2018
    Our contribution: combining these ideas with selective SSMs.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .config import SSMConfig
from .mamba import MambaLayer

logger = logging.getLogger(__name__)


class PhysicsSSM(nn.Module):
    """
    Physics-guided SSM where hidden state evolution is influenced
    by physics simulation constraints through a learned gate:

        gate = sigmoid(W [ssm_out ; physics_embed])
        output = gate * ssm_out + (1 - gate) * constraints
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 32,
        d_physics: int = 64,
        gravity: float = -9.81,
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
        physics_state: Optional[torch.Tensor] = None,
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

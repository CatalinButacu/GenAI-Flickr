
from __future__ import annotations


import torch
import torch.nn as nn

from src.modules.motion.ssm import MambaLayer, SSMConfig


class MotionProjector(nn.Module):
    """Project d_model features → motion_dim and back."""

    def __init__(self, d_model: int, motion_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(),
            nn.Linear(d_model, motion_dim),
        )

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        return self.encoder(motion)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


class SimpleTextEncoder(nn.Module):
    """Word-embedding + TransformerEncoder, pooled to (batch, embed_dim)."""

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed  = nn.Embedding(max_length, embed_dim)
        self.encoder    = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True),
            num_layers=2,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, s   = token_ids.shape
        pos    = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, -1)
        x      = self.word_embed(token_ids) + self.pos_embed(pos)
        x      = self.encoder(x)
        return x.mean(dim=1)


class MotionDecoder(nn.Module):
    """Projects SSM features → (motion, predicted_length)."""

    def __init__(self, d_model: int, motion_dim: int, max_length: int):
        super().__init__()
        self.max_length  = max_length
        self.length_head = nn.Linear(d_model, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, motion_dim),
        )

    def forward(self, features: torch.Tensor,
                condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length_pred = (
            torch.sigmoid(self.length_head(condition)).squeeze(-1)
            * self.max_length
        )
        return self.output_proj(features), length_pred


class TextToMotionSSM(nn.Module):
    """Full text-to-motion model: TextEncoder → SSM layers → MotionDecoder."""

    def __init__(self, config):
        """config: TrainingConfig (from trainer module)."""
        super().__init__()
        self.config        = config
        self.text_encoder  = SimpleTextEncoder(
            config.vocab_size, config.text_embed_dim, config.max_text_length,
        )
        self.condition_proj = nn.Linear(config.text_embed_dim, config.d_model)
        self.pos_embed      = nn.Embedding(config.max_motion_length, config.d_model)
        ssm_cfg  = SSMConfig(d_model=config.d_model, d_state=config.d_state)
        self.layers = nn.ModuleList([MambaLayer(ssm_cfg) for _ in range(config.n_layers)])
        self.norms  = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.n_layers)])
        self.decoder = MotionDecoder(config.d_model, config.motion_dim, config.max_motion_length)

    def forward(self, token_ids: torch.Tensor,
                motion_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cond = self.condition_proj(self.text_encoder(token_ids))
        if motion_length is None:
            motion_length = self.config.max_motion_length
        pos = self.pos_embed(torch.arange(motion_length, device=token_ids.device))
        x   = cond.unsqueeze(1) + pos.unsqueeze(0)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(norm(x))
        return self.decoder(x, cond)

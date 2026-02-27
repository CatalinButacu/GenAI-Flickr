"""SSM text-to-motion training: TextToMotionSSM trained on KIT-ML."""

import os
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from .constants import (
    DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT, MOTION_DIM, MOTION_FPS,
)
from .tokenizer import build_vocab, tokenize

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    log.warning("PyTorch not available")


@dataclass
class TrainingConfig:
    data_dir: str            = DEFAULT_DATA_DIR
    max_motion_length: int   = 200
    d_model: int             = 256
    d_state: int             = 32
    n_layers: int            = 4
    motion_dim: int          = MOTION_DIM
    text_embed_dim: int      = 256
    max_text_length: int     = 64
    vocab_size: int          = 10000
    batch_size: int          = 32
    learning_rate: float     = 1e-4
    weight_decay: float      = 0.01
    num_epochs: int          = 100
    warmup_steps: int        = 1000
    grad_clip: float         = 1.0
    checkpoint_dir: str      = os.path.dirname(DEFAULT_SSM_CHECKPOINT)
    save_every: int          = 10
    device: str              = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"


if HAS_TORCH:

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
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            b, s   = token_ids.shape
            pos    = torch.arange(s, device=token_ids.device).unsqueeze(0).expand(b, -1)
            x      = self.word_embed(token_ids) + self.pos_embed(pos)
            x      = self.encoder(x)
            return self.pool(x.transpose(1, 2)).squeeze(-1)


    class MotionDecoder(nn.Module):
        """Projects SSM features  (motion, predicted_length)."""
        def __init__(self, d_model: int, motion_dim: int, max_length: int):
            super().__init__()
            self.max_length  = max_length
            self.length_head = nn.Linear(d_model, 1)
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, motion_dim)
            )

        def forward(self, features: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            length_pred = torch.sigmoid(self.length_head(condition)).squeeze(-1) * self.max_length
            return self.output_proj(features), length_pred


    class TextToMotionSSM(nn.Module):
        """Full text-to-motion model: TextEncoder  SSM layers  MotionDecoder."""
        def __init__(self, config: TrainingConfig):
            super().__init__()
            self.config        = config
            self.text_encoder  = SimpleTextEncoder(config.vocab_size, config.text_embed_dim, config.max_text_length)
            self.condition_proj = nn.Linear(config.text_embed_dim, config.d_model)
            self.pos_embed     = nn.Embedding(config.max_motion_length, config.d_model)
            from src.ssm import MambaLayer, SSMConfig
            ssm_cfg   = SSMConfig(d_model=config.d_model, d_state=config.d_state)
            self.layers = nn.ModuleList([MambaLayer(ssm_cfg) for _ in range(config.n_layers)])
            self.norms  = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in range(config.n_layers)])
            self.decoder = MotionDecoder(config.d_model, config.motion_dim, config.max_motion_length)

        def forward(self, token_ids: torch.Tensor, motion_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
            cond = self.condition_proj(self.text_encoder(token_ids))
            if motion_length is None:
                motion_length = self.config.max_motion_length
            pos = self.pos_embed(torch.arange(motion_length, device=token_ids.device))
            x   = cond.unsqueeze(1) + pos.unsqueeze(0)
            for layer, norm in zip(self.layers, self.norms):
                x = x + layer(norm(x))
            return self.decoder(x, cond)


    class KITMLDatasetTorch(Dataset):
        """KIT-ML PyTorch Dataset: tokenized text + padded motion + mask."""
        def __init__(self, data_dir: str, split: str = "train",
                     max_length: int = 200, vocab: Dict[str, int] = None):
            from src.data import KITMLLoader
            self.dataset    = KITMLLoader(data_dir).load_dataset(split, normalize=True)
            self.max_length = max_length
            self.vocab      = vocab or build_vocab([s.text for s in self.dataset.samples])
            if vocab is None:
                log.info("Vocab size: %d", len(self.vocab))

        def __len__(self):
            return len(self.dataset.samples)

        def __getitem__(self, idx):
            s      = self.dataset.samples[idx]
            tokens = tokenize(s.text, self.vocab)
            motion = s.motion[:self.max_length]
            if len(motion) < self.max_length:
                motion = np.concatenate([motion, np.zeros((self.max_length - len(motion), motion.shape[1]))])
            mask        = np.zeros(self.max_length)
            mask[:len(s.motion)] = 1.0
            return {
                "token_ids":    torch.tensor(tokens, dtype=torch.long),
                "motion":       torch.tensor(motion, dtype=torch.float32),
                "motion_mask":  torch.tensor(mask,   dtype=torch.float32),
                "length":       len(s.motion),
                "text":         s.text,
            }


    class Trainer:
        """AdamW + OneCycleLR training loop for TextToMotionSSM."""
        def __init__(self, config: TrainingConfig):
            self.config = config
            self.device = torch.device(config.device)
            self.train_ds   = KITMLDatasetTorch(config.data_dir, "train", config.max_motion_length)
            self.val_ds     = KITMLDatasetTorch(config.data_dir, "val",   config.max_motion_length, vocab=self.train_ds.vocab)
            config.vocab_size = len(self.train_ds.vocab)
            self.model      = TextToMotionSSM(config).to(self.device)
            self.train_loader = DataLoader(self.train_ds, config.batch_size, shuffle=True,  num_workers=0)
            self.val_loader   = DataLoader(self.val_ds,   config.batch_size, shuffle=False, num_workers=0)
            self.optimizer  = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            total_steps     = max(len(self.train_loader) * config.num_epochs, 1)
            warmup_pct      = min(config.warmup_steps / total_steps, 0.3)
            self.scheduler  = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=config.learning_rate, total_steps=total_steps, pct_start=warmup_pct)
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            self.step       = 0
            self.best_loss  = float("inf")

        def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
            self.model.train()
            tot = mot = lng = 0.0
            try:
                from tqdm import tqdm
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
            except ImportError:
                pbar = self.train_loader
            for batch in pbar:
                tkn  = batch["token_ids"].to(self.device)
                mgt  = batch["motion"].to(self.device)
                msk  = batch["motion_mask"].to(self.device)
                pred, lp = self.model(tkn, mgt.shape[1])
                ml   = F.mse_loss(pred * msk.unsqueeze(-1), mgt * msk.unsqueeze(-1))
                ll   = F.mse_loss(lp, batch["length"].float().to(self.device))
                loss = ml + 0.1 * ll
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                tot += loss.item();  mot += ml.item();  lng += ll.item()
                self.step += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"loss": f"{loss.item():.3f}", "mot": f"{ml.item():.3f}"})
                if self.step % 50 == 0:
                    log.info("step=%d loss=%.4f mot=%.4f lr=%.2e", self.step, loss.item(), ml.item(), self.optimizer.param_groups[0]["lr"])
            n = len(self.train_loader)
            return tot / n, mot / n, lng / n

        def _validate(self) -> float:
            self.model.eval()
            total = 0.0
            with torch.no_grad():
                for batch in self.val_loader:
                    tkn  = batch["token_ids"].to(self.device)
                    mgt  = batch["motion"].to(self.device)
                    msk  = batch["motion_mask"].to(self.device)
                    pred, _ = self.model(tkn, mgt.shape[1])
                    total += F.mse_loss(pred * msk.unsqueeze(-1), mgt * msk.unsqueeze(-1)).item()
            return total / len(self.val_loader)

        def _save(self, epoch: int, val_loss: float, is_best: bool) -> None:
            ck = {
                "epoch": epoch, "global_step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss, "config": self.config,
                "vocab": self.train_ds.vocab,
            }
            torch.save(ck, os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"))
            if is_best:
                torch.save(ck, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
                log.info("Best model saved: val_loss=%.4f", val_loss)

        def train(self) -> float:
            log.info("device=%s train=%d val=%d vocab=%d params=%s",
                     self.device, len(self.train_ds), len(self.val_ds),
                     self.config.vocab_size,
                     f"{sum(p.numel() for p in self.model.parameters()):,}")
            for epoch in range(1, self.config.num_epochs + 1):
                tl, ml, ll = self._train_epoch(epoch)
                vl = self._validate()
                log.info("epoch=%d/%d train=%.4f(mot=%.4f len=%.2f) val=%.4f lr=%.2e",
                         epoch, self.config.num_epochs, tl, ml, ll, vl,
                         self.optimizer.param_groups[0]["lr"])
                is_best = vl < self.best_loss
                if is_best:
                    self.best_loss = vl
                if epoch % self.config.save_every == 0 or is_best:
                    self._save(epoch, vl, is_best)
            return self.best_loss


def train_motion_ssm(config: TrainingConfig = None) -> float:
    if not HAS_TORCH:
        raise RuntimeError("PyTorch required for training")
    return Trainer(config or TrainingConfig()).train()

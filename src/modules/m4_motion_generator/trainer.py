"""
#WHERE
    Used by scripts/train_motion_ssm.py to launch SSM training.

#WHAT
    AdamW + OneCycleLR training loop for TextToMotionSSM.
    Contains TrainingConfig (hyperparameters) and the Trainer class
    that orchestrates epoch-level train/validate/save cycles.

#INPUT
    TrainingConfig with data paths, model hyper-params, optimiser settings.

#OUTPUT
    Best validation loss (float).  Checkpoints saved to checkpoint_dir.
"""

import os
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT, MOTION_DIM
from .dataset import KITMLDatasetTorch
from .nn_models import TextToMotionSSM

log = logging.getLogger(__name__)


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
    device: str              = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """AdamW + OneCycleLR training loop for TextToMotionSSM."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.train_ds   = KITMLDatasetTorch(config.data_dir, "train", config.max_motion_length)
        self.val_ds     = KITMLDatasetTorch(config.data_dir, "val",   config.max_motion_length,
                                            vocab=self.train_ds.vocab)
        config.vocab_size = len(self.train_ds.vocab)

        self.model        = TextToMotionSSM(config).to(self.device)
        self.train_loader = DataLoader(self.train_ds, config.batch_size, shuffle=True,  num_workers=0)
        self.val_loader   = DataLoader(self.val_ds,   config.batch_size, shuffle=False, num_workers=0)

        self.optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        total_steps = max(len(self.train_loader) * config.num_epochs, 1)
        warmup_pct  = min(config.warmup_steps / total_steps, 0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=warmup_pct,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.step      = 0
        self.best_loss = float("inf")

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        tot = mot = lng = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
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
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "mot": f"{ml.item():.3f}"})
            if self.step % 50 == 0:
                log.info("step=%d loss=%.4f mot=%.4f lr=%.2e",
                         self.step, loss.item(), ml.item(),
                         self.optimizer.param_groups[0]["lr"])
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

    # ── Main loop ────────────────────────────────────────────────────────────

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
    """Entry point for training the text-to-motion SSM."""
    return Trainer(config or TrainingConfig()).train()

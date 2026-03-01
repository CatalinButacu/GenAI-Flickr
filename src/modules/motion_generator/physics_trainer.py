"""
#WHERE
    Used by scripts/train_physics_ssm.py to train the PhysicsSSM model.

#WHAT
    Training loop for the PhysicsSSM — the novel contribution that blends
    SSM temporal modelling with physics constraints via a learned sigmoid gate:

        gate = σ(W [ssm_out ; physics_embed])
        output = gate · ssm_out + (1 - gate) · constraints

    The training objective has two components:
      L_total = L_reconstruction + λ · L_physics

      L_reconstruction: MSE between PhysicsSSM-refined motion and target motion.
      L_physics: physics violation penalty — penalises foot sliding, ground
                 penetration, and non-physical accelerations.

#INPUT
    PhysicsTrainingConfig with data paths, hyperparameters.

#OUTPUT
    Best validation loss (float).  Checkpoint saved to checkpoint_dir.
"""

import os
import logging
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import DEFAULT_DATA_DIR, MOTION_DIM
from .physics_dataset import PhysicsMotionDataset, D_PHYSICS
from .ssm.composites import MotionSSM, PhysicsSSM

log = logging.getLogger(__name__)


from src.shared.constants import DEFAULT_PHYSICS_SSM_CHECKPOINT


@dataclass
class PhysicsTrainingConfig:
    data_dir: str            = DEFAULT_DATA_DIR
    max_motion_length: int   = 200
    d_model: int             = 256
    d_state: int             = 32
    d_physics: int           = D_PHYSICS
    n_layers: int            = 4
    motion_dim: int          = MOTION_DIM
    batch_size: int          = 16
    learning_rate: float     = 5e-5
    weight_decay: float      = 0.01
    num_epochs: int          = 100
    warmup_steps: int        = 500
    grad_clip: float         = 0.5
    lambda_physics: float    = 0.1       # weight for physics violation loss
    resume_from: str         = ""        # path to checkpoint to resume from
    checkpoint_dir: str      = os.path.dirname(DEFAULT_PHYSICS_SSM_CHECKPOINT)
    save_every: int          = 10
    patience: int            = 15        # early stopping patience (0 = disabled)
    device: str              = "cuda" if torch.cuda.is_available() else "cpu"


class MotionProjector(nn.Module):
    """Project d_model features → motion_dim and back.

    Replaces the random projection in ssm_generator with a learned one.
    """

    def __init__(self, d_model: int, motion_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, motion_dim),
        )

    def encode(self, motion: torch.Tensor) -> torch.Tensor:
        """(batch, T, motion_dim) → (batch, T, d_model)."""
        return self.encoder(motion)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """(batch, T, d_model) → (batch, T, motion_dim)."""
        return self.decoder(features)


class PhysicsSSMTrainer:
    """Training loop for MotionSSM + PhysicsSSM with physics-aware loss."""

    def __init__(self, config: PhysicsTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Data
        self.train_ds = PhysicsMotionDataset(
            config.data_dir, "train", config.max_motion_length, config.d_physics,
        )
        self.val_ds = PhysicsMotionDataset(
            config.data_dir, "val", config.max_motion_length, config.d_physics,
        )

        self.train_loader = DataLoader(
            self.train_ds, config.batch_size, shuffle=True, num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_ds, config.batch_size, shuffle=False, num_workers=0,
        )

        # Models
        self.motion_ssm = MotionSSM(
            d_model=config.d_model, d_state=config.d_state, n_layers=config.n_layers,
        ).to(self.device)

        self.physics_ssm = PhysicsSSM(
            d_model=config.d_model, d_state=config.d_state, d_physics=config.d_physics,
        ).to(self.device)

        self.projector = MotionProjector(
            d_model=config.d_model, motion_dim=config.motion_dim,
        ).to(self.device)

        # Combined parameters
        all_params = (
            list(self.motion_ssm.parameters())
            + list(self.physics_ssm.parameters())
            + list(self.projector.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )

        total_steps = max(len(self.train_loader) * config.num_epochs, 1)
        warmup_pct = min(config.warmup_steps / total_steps, 0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=warmup_pct,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.step = 0
        self.start_epoch = 1
        self.best_loss = float("inf")

        if config.resume_from and os.path.isfile(config.resume_from):
            self._load_resume(config.resume_from)

    # ── Resume from checkpoint ──────────────────────────────────────────

    def _load_resume(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.motion_ssm.load_state_dict(ck["motion_ssm_state_dict"])
        self.physics_ssm.load_state_dict(ck["physics_ssm_state_dict"])
        self.projector.load_state_dict(ck["projector_state_dict"])
        self.best_loss = ck.get("val_loss", float("inf"))
        self.step = ck.get("global_step", 0)
        self.start_epoch = ck.get("epoch", 0) + 1
        # Recreate scheduler for remaining steps (no warmup — model is past that)
        remaining = max(self.config.num_epochs - self.start_epoch + 1, 1)
        remaining_steps = max(len(self.train_loader) * remaining, 1)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=remaining_steps, eta_min=1e-6,
        )
        log.info(
            "Resumed from %s — start_epoch=%d best_loss=%.4f",
            path, self.start_epoch, self.best_loss,
        )

    # ── Loss functions ───────────────────────────────────────────────────

    @staticmethod
    def _physics_violation_loss(
        pred_motion: torch.Tensor,
        physics_state: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute physics violation penalties.

        Penalises:
          1. **Foot sliding**: when foot contacts are active (physics_state[:,:,10:12]),
             the motion velocity should be near zero.
          2. **Ground penetration**: pelvis height (physics_state[:,:,1]) should
             not go below a minimum threshold.
          3. **Jerk penalty**: penalise high-frequency jitter (3rd derivative).

        Args:
            pred_motion: (B, T, motion_dim) predicted motion
            physics_state: (B, T, d_physics) physics state
            mask: (B, T) validity mask
        """
        B, T, D = pred_motion.shape
        mask_3d = mask.unsqueeze(-1)  # (B, T, 1)

        losses = []

        # 1. Foot sliding penalty — velocity during contact should be low
        # Approximate velocity from finite differences of root position
        if T > 1:
            vel = pred_motion[:, 1:, 3:6] - pred_motion[:, :-1, 3:6]  # root vel diff
            # Foot contacts from physics state — clamp to [0,1] (binary; normalization
            # can introduce negatives which would flip the penalty into a reward)
            foot_contact = physics_state[:, 1:, 10:12].mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
            slide_loss = (vel.pow(2) * foot_contact * mask[:, 1:].unsqueeze(-1)).mean()
            losses.append(slide_loss)

        # 2. Ground penetration — penalise negative pelvis height
        pelvis_h = physics_state[:, :, 1]  # estimated pelvis height
        penetration = F.relu(-pelvis_h) * mask  # only penalise below 0
        losses.append(penetration.pow(2).mean())

        # 3. Jerk penalty — smooth motion (penalise 3rd derivative)
        if T > 3:
            # 1st diff = velocity, 2nd = accel, 3rd = jerk
            d1 = pred_motion[:, 1:] - pred_motion[:, :-1]
            d2 = d1[:, 1:] - d1[:, :-1]
            d3 = d2[:, 1:] - d2[:, :-1]
            jerk_mask = mask[:, 3:].unsqueeze(-1)
            jerk_loss = (d3.pow(2) * jerk_mask).mean() * 0.01  # downweight
            losses.append(jerk_loss)

        phys = sum(losses) if losses else torch.tensor(0.0, device=pred_motion.device)
        # Safety clamp: physics violation penalties must be non-negative
        return phys.clamp(min=0.0)

    # ── Training loop ────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.motion_ssm.train()
        self.physics_ssm.train()
        self.projector.train()

        tot_loss = 0.0
        tot_recon = 0.0
        tot_phys = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
        for batch in pbar:
            motion = batch["motion"].to(self.device)          # (B, T, 251)
            physics = batch["physics_state"].to(self.device)  # (B, T, 64)
            mask = batch["motion_mask"].to(self.device)       # (B, T)
            mask_3d = mask.unsqueeze(-1)                      # (B, T, 1)

            # Forward: motion → d_model → SSM → PhysicsSSM → d_model → motion
            encoded = self.projector.encode(motion)            # (B, T, d_model)
            ssm_out = self.motion_ssm(encoded)                 # (B, T, d_model)
            phys_out = self.physics_ssm(ssm_out, physics)      # (B, T, d_model)
            decoded = self.projector.decode(phys_out)           # (B, T, 251)

            # L_reconstruction: MSE on valid frames
            recon_loss = F.mse_loss(
                decoded * mask_3d, motion * mask_3d,
            )

            # L_physics: violation penalties
            phys_loss = self._physics_violation_loss(decoded, physics, mask)

            loss = recon_loss + self.config.lambda_physics * phys_loss

            # Guard against NaN/Inf (can occur with extreme physics state values)
            if torch.isnan(loss) or torch.isinf(loss):
                log.warning("NaN/Inf loss at step %d — skipping batch", self.step)
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.motion_ssm.parameters())
                + list(self.physics_ssm.parameters())
                + list(self.projector.parameters()),
                self.config.grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()

            tot_loss += loss.item()
            tot_recon += recon_loss.item()
            tot_phys += phys_loss.item()
            self.step += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "phys": f"{phys_loss.item():.4f}",
            })

            if self.step % 50 == 0:
                log.info(
                    "step=%d loss=%.4f recon=%.4f phys=%.4f lr=%.2e",
                    self.step, loss.item(), recon_loss.item(), phys_loss.item(),
                    self.optimizer.param_groups[0]["lr"],
                )

        n = max(len(self.train_loader), 1)
        return tot_loss / n, tot_recon / n, tot_phys / n

    @torch.no_grad()
    def _validate(self) -> Tuple[float, float, float]:
        self.motion_ssm.eval()
        self.physics_ssm.eval()
        self.projector.eval()

        tot_loss = 0.0
        tot_recon = 0.0
        tot_phys = 0.0

        for batch in self.val_loader:
            motion = batch["motion"].to(self.device)
            physics = batch["physics_state"].to(self.device)
            mask = batch["motion_mask"].to(self.device)
            mask_3d = mask.unsqueeze(-1)

            encoded = self.projector.encode(motion)
            ssm_out = self.motion_ssm(encoded)
            phys_out = self.physics_ssm(ssm_out, physics)
            decoded = self.projector.decode(phys_out)

            recon_loss = F.mse_loss(decoded * mask_3d, motion * mask_3d)
            phys_loss = self._physics_violation_loss(decoded, physics, mask)
            loss = recon_loss + self.config.lambda_physics * phys_loss

            tot_loss += loss.item()
            tot_recon += recon_loss.item()
            tot_phys += phys_loss.item()

        n = max(len(self.val_loader), 1)
        return tot_loss / n, tot_recon / n, tot_phys / n

    def _save(self, epoch: int, val_loss: float, is_best: bool) -> None:
        ck = {
            "epoch": epoch,
            "global_step": self.step,
            "motion_ssm_state_dict": self.motion_ssm.state_dict(),
            "physics_ssm_state_dict": self.physics_ssm.state_dict(),
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt",
        )
        torch.save(ck, path)
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            torch.save(ck, best_path)
            log.info("Best model saved: val_loss=%.4f -> %s", val_loss, best_path)

    def train(self) -> float:
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.motion_ssm, self.physics_ssm, self.projector]
        )
        log.info(
            "device=%s train=%d val=%d params=%s",
            self.device, len(self.train_ds), len(self.val_ds),
            f"{total_params:,}",
        )
        log.info(
            "config: d_model=%d d_state=%d d_physics=%d n_layers=%d "
            "lambda_physics=%.2f lr=%.1e epochs=%d",
            self.config.d_model, self.config.d_state, self.config.d_physics,
            self.config.n_layers, self.config.lambda_physics,
            self.config.learning_rate, self.config.num_epochs,
        )

        no_improve = 0
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            tl, tr, tp = self._train_epoch(epoch)
            vl, vr, vp = self._validate()

            log.info(
                "epoch=%d/%d train=%.4f(recon=%.4f phys=%.4f) "
                "val=%.4f(recon=%.4f phys=%.4f) lr=%.2e",
                epoch, self.config.num_epochs, tl, tr, tp,
                vl, vr, vp,
                self.optimizer.param_groups[0]["lr"],
            )

            is_best = vl < self.best_loss
            if is_best:
                self.best_loss = vl
                no_improve = 0
            else:
                no_improve += 1
            if epoch % self.config.save_every == 0 or is_best:
                self._save(epoch, vl, is_best)

            # Early stopping
            if self.config.patience > 0 and no_improve >= self.config.patience:
                log.info(
                    "Early stopping at epoch %d — no improvement for %d epochs "
                    "(best_val_loss=%.4f)",
                    epoch, self.config.patience, self.best_loss,
                )
                break

        return self.best_loss


def train_physics_ssm(config: PhysicsTrainingConfig = None) -> float:
    """Entry point for training the PhysicsSSM."""
    return PhysicsSSMTrainer(config or PhysicsTrainingConfig()).train()

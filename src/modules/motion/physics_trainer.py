
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import DEFAULT_DATA_DIR, MOTION_DIM, D_PHYSICS
from src.shared.data.physics_dataset import PhysicsMotionDataset
from .ssm.composites import MotionSSM, PhysicsSSM
from src.shared.constants import (
    DEFAULT_PHYSICS_SSM_CHECKPOINT, SSM_D_MODEL, SSM_D_STATE, SSM_N_LAYERS,
)

log = logging.getLogger(__name__)


@dataclass
class PhysicsTrainingConfig:
    data_dir: str            = DEFAULT_DATA_DIR
    # Same 200-frame cap as MotionSSM — see TrainingConfig comment.
    max_motion_length: int   = 200
    d_model: int             = SSM_D_MODEL     # 256 (paper §3)
    d_state: int             = SSM_D_STATE     # 32  (paper §3)
    d_physics: int           = D_PHYSICS       # 64  — physics state channels (architectural)
    n_layers: int            = SSM_N_LAYERS    # 4   (paper §3)
    motion_dim: int          = MOTION_DIM      # 168 (architectural: SMPL-X)
    # HARDWARE CHOICE: 16 because PhysicsSSM processes motion+physics tensors
    # (168+64=232 dim per frame) — roughly 2× memory per sample vs MotionSSM.
    # Paper §4: "batch size 16".
    batch_size: int          = 16
    # TUNING CHOICE: 5e-5 is lower than MotionSSM's 1e-4 because PhysicsSSM
    # has a more complex loss landscape (motion + physics gating). Paper §4: "lr=5e-5".
    learning_rate: float     = 5e-5
    weight_decay: float      = 0.01   # standard AdamW regularisation
    num_epochs: int          = 100    # paper §4: "100 epochs"
    warmup_steps: int        = 500    # smaller dataset → fewer warmup steps than MotionSSM
    # TUNING CHOICE: 0.5 is stricter than MotionSSM's 1.0 because the physics
    # gate (sigmoid) is sensitive to gradient spikes. Empirically prevents NaN.
    grad_clip: float         = 0.5
    # DESIGN CHOICE: weight for physics violation penalty in the composite loss.
    # lambda=0.1 means motion reconstruction dominates; physics is a regulariser.
    # Paper §5: "ablation shows lambda=0.1 optimal" (Table 2).
    lambda_physics: float    = 0.1
    resume_from: str         = ""
    checkpoint_dir: str      = os.path.dirname(DEFAULT_PHYSICS_SSM_CHECKPOINT)
    save_every: int          = 10
    # DESIGN CHOICE: stop training if val loss doesn't improve for 15 epochs.
    # With 100 total epochs, this prevents wasting ~85% of training on overfitting.
    patience: int            = 15
    device: str              = "cuda" if torch.cuda.is_available() else "cpu"
    accum_steps: int         = 1      # gradient accumulation (effective batch = batch_size × accum)


from src.modules.motion.nn_models import MotionProjector  # noqa: F401 — re-export for backward compat


class PhysicsSSMTrainer:
    """Training loop for MotionSSM + PhysicsSSM with physics-aware loss."""

    def __init__(self, config: PhysicsTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.train_ds, self.val_ds = self._create_datasets(config)
        self.train_loader, self.val_loader = self._create_loaders(config)
        self.motion_ssm, self.physics_ssm, self.projector = self._create_models(config)
        self.optimizer, self.scheduler = self._create_optimizer(config)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.step = 0
        self.start_epoch = 1
        self.best_loss = float("inf")

        if config.resume_from and os.path.isfile(config.resume_from):
            self.load_resume(config.resume_from)

    def _create_datasets(self, config):
        """Initialise train and validation datasets with proper split.

        Loads data once, then splits 80/20 by index so train and val
        never share the same sequences (prevents data leakage).
        """
        # Load all samples once (no augmentation yet)
        full_ds = PhysicsMotionDataset(
            data_dir=config.data_dir,
            max_length=config.max_motion_length,
            d_physics=config.d_physics,
            augment=False,
        )
        all_samples = full_ds._samples

        # Deterministic shuffle + 80/20 split
        import random
        indices = list(range(len(all_samples)))
        random.Random(42).shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices = indices[:split]
        val_indices = indices[split:]

        train_samples = [all_samples[i] for i in train_indices]
        val_samples = [all_samples[i] for i in val_indices]

        log.info("[Trainer] data split: %d train / %d val (from %d total)",
                 len(train_samples), len(val_samples), len(all_samples))

        train_ds = PhysicsMotionDataset(
            max_length=config.max_motion_length,
            d_physics=config.d_physics,
            augment=True,
            _preloaded=train_samples,
        )
        # Val uses training set's normalization stats (no data leakage)
        val_ds = PhysicsMotionDataset(
            max_length=config.max_motion_length,
            d_physics=config.d_physics,
            augment=False,
            _preloaded=val_samples,
        )
        val_ds._phys_mean = train_ds._phys_mean
        val_ds._phys_std = train_ds._phys_std
        return train_ds, val_ds

    def _create_loaders(self, config):
        """Create DataLoader instances for train and val splits."""
        # Use multiple workers on Linux/Mac; Windows requires 0 for safety
        import sys
        n_workers = min(4, os.cpu_count() or 0) if sys.platform != "win32" else 0
        train_loader = DataLoader(
            self.train_ds, config.batch_size, shuffle=True,
            num_workers=n_workers, persistent_workers=n_workers > 0,
        )
        val_loader = DataLoader(
            self.val_ds, config.batch_size, shuffle=False,
            num_workers=n_workers, persistent_workers=n_workers > 0,
        )
        return train_loader, val_loader

    def _create_models(self, config):
        """Instantiate MotionSSM, PhysicsSSM, and MotionProjector."""
        motion_ssm = MotionSSM(
            d_model=config.d_model, d_state=config.d_state, n_layers=config.n_layers,
        ).to(self.device)
        physics_ssm = PhysicsSSM(
            d_model=config.d_model, d_state=config.d_state, d_physics=config.d_physics,
        ).to(self.device)
        projector = MotionProjector(
            d_model=config.d_model, motion_dim=config.motion_dim,
        ).to(self.device)
        return motion_ssm, physics_ssm, projector

    def _create_optimizer(self, config):
        """Set up AdamW optimizer with OneCycleLR scheduler."""
        all_params = (
            list(self.motion_ssm.parameters())
            + list(self.physics_ssm.parameters())
            + list(self.projector.parameters())
        )
        optimizer = torch.optim.AdamW(
            all_params, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        total_steps = max(len(self.train_loader) * config.num_epochs, 1)
        warmup_pct = min(config.warmup_steps / total_steps, 0.3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=warmup_pct,
        )
        return optimizer, scheduler

    # ── Resume from checkpoint ──────────────────────────────────────────

    def load_resume(self, path: str) -> None:
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
    def physics_violation_loss(
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

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.motion_ssm.train()
        self.physics_ssm.train()
        self.projector.train()

        tot_loss = 0.0
        tot_recon = 0.0
        tot_phys = 0.0
        accum = self.config.accum_steps
        all_params = self._all_trainable_params()

        self.optimizer.zero_grad()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
        for i, batch in enumerate(pbar):
            recon_loss, phys_loss, loss = self._forward_batch(batch, accum)
            if loss is None:
                continue

            loss.backward()
            scaled = loss.item() * accum
            tot_loss += scaled
            tot_recon += recon_loss.item()
            tot_phys += phys_loss.item()

            self._maybe_optimizer_step(i, all_params, scaled, recon_loss, phys_loss)
            pbar.set_postfix(loss=f"{scaled:.4f}",
                             recon=f"{recon_loss.item():.4f}",
                             phys=f"{phys_loss.item():.4f}")

        n = max(len(self.train_loader), 1)
        return tot_loss / n, tot_recon / n, tot_phys / n

    def _all_trainable_params(self):
        """Collect parameters from all sub-models."""
        return (list(self.motion_ssm.parameters())
                + list(self.physics_ssm.parameters())
                + list(self.projector.parameters()))

    def _forward_batch(self, batch, accum):
        """Run forward pass on one batch, return (recon_loss, phys_loss, scaled_loss)."""
        motion = batch["motion"].to(self.device)
        physics = batch["physics_state"].to(self.device)
        mask = batch["motion_mask"].to(self.device)
        mask_3d = mask.unsqueeze(-1)

        encoded = self.projector.encode(motion)
        ssm_out = self.motion_ssm(encoded)
        phys_out = self.physics_ssm(ssm_out, physics)
        decoded = self.projector.decode(phys_out)

        recon_loss = F.mse_loss(decoded * mask_3d, motion * mask_3d)
        phys_loss = self.physics_violation_loss(decoded, physics, mask)
        loss = (recon_loss + self.config.lambda_physics * phys_loss) / accum

        if torch.isnan(loss) or torch.isinf(loss):
            log.warning("NaN/Inf loss at step %d — skipping batch", self.step)
            self.optimizer.zero_grad()
            return recon_loss, phys_loss, None
        return recon_loss, phys_loss, loss

    def _maybe_optimizer_step(self, batch_idx, all_params, scaled, recon_loss, phys_loss):
        """Perform optimizer step if accumulation window is complete."""
        accum = self.config.accum_steps
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(self.train_loader):
            torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.step += 1
            if self.step % 50 == 0:
                log.info(
                    "step=%d loss=%.4f recon=%.4f phys=%.4f lr=%.2e",
                    self.step, scaled, recon_loss.item(), phys_loss.item(),
                    self.optimizer.param_groups[0]["lr"],
                )

    @torch.no_grad()
    def validate(self) -> tuple[float, float, float]:
        torch.cuda.empty_cache()
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
            phys_loss = self.physics_violation_loss(decoded, physics, mask)
            loss = recon_loss + self.config.lambda_physics * phys_loss

            tot_loss += loss.item()
            tot_recon += recon_loss.item()
            tot_phys += phys_loss.item()

        n = max(len(self.val_loader), 1)
        return tot_loss / n, tot_recon / n, tot_phys / n

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
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
        self._log_training_config()

        no_improve = 0
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            tl, tr, tp = self.train_epoch(epoch)
            vl, vr, vp = self.validate()

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
                self.save_checkpoint(epoch, vl, is_best)

            if self._should_early_stop(epoch, no_improve):
                break

        return self.best_loss

    def _log_training_config(self) -> None:
        """Log model and training configuration at start."""
        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.motion_ssm, self.physics_ssm, self.projector]
        )
        log.info("device=%s train=%d val=%d params=%s",
                 self.device, len(self.train_ds), len(self.val_ds),
                 f"{total_params:,}")
        log.info("config: d_model=%d d_state=%d d_physics=%d n_layers=%d "
                 "lambda_physics=%.2f lr=%.1e epochs=%d",
                 self.config.d_model, self.config.d_state, self.config.d_physics,
                 self.config.n_layers, self.config.lambda_physics,
                 self.config.learning_rate, self.config.num_epochs)

    def _should_early_stop(self, epoch: int, no_improve: int) -> bool:
        """Check early stopping condition."""
        if self.config.patience > 0 and no_improve >= self.config.patience:
            log.info("Early stopping at epoch %d — no improvement for %d epochs "
                     "(best_val_loss=%.4f)", epoch, self.config.patience, self.best_loss)
            return True
        return False


def train_physics_ssm(config: PhysicsTrainingConfig = None) -> float:
    """Entry point for training the PhysicsSSM."""
    return PhysicsSSMTrainer(config or PhysicsTrainingConfig()).train()

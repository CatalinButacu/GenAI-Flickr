
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
import glob
import re

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT, MOTION_DIM
from src.shared.data.motion_dataset import MotionDataset
from .nn_models import TextToMotionSSM
from src.shared.constants import SSM_D_MODEL, SSM_D_STATE, SSM_N_LAYERS

log = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    """Lock all RNG sources for fully reproducible training runs.

    Must be called BEFORE model initialisation, DataLoader creation,
    and optimizer creation to guarantee identical results across runs.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # deterministic=True guarantees bitwise reproducibility at the cost of
        # some speed. benchmark=False disables the auto-tuner (also non-deterministic).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


@dataclass
class TrainingConfig:
    data_dir: str            = DEFAULT_DATA_DIR
    # DATASET CONSTRAINT: 200 frames = 6.67s at 30fps. AMASS sequences average
    # 3-8s; longer sequences are truncated, shorter are zero-padded. This caps
    # GPU memory per sample (~200×168 floats = 134KB). Paper §4: "200 frames".
    max_motion_length: int   = 200
    d_model: int             = SSM_D_MODEL     # 256 — Mamba hidden dim (paper §3: d_model=256)
    d_state: int             = SSM_D_STATE     # 32  — SSM state size (paper §3: d_state=32)
    n_layers: int            = SSM_N_LAYERS    # 4   — Mamba layers (paper §3: 4 layers)
    motion_dim: int          = MOTION_DIM      # 168 — SMPL-X axis-angle pose dim (architectural)
    # DESIGN CHOICE: text embedding before conditioning projection.
    # 256 matches d_model so condition_proj is a square linear layer.
    text_embed_dim: int      = 256
    # DESIGN CHOICE: 64 tokens ≈ ~50 words. Motion text descriptions are short
    # (AMASS has none, Inter-X averages ~25 words). Increase if using longer descriptions.
    max_text_length: int     = 64
    # ARBITRARY: initial vocab capacity. Overwritten in Trainer.__init__ after
    # build_vocab() counts the actual vocabulary from training data.
    vocab_size: int          = 10000
    # HARDWARE CHOICE: 32 fits in ~6GB VRAM. Paper §4: "batch size 16" (trained
    # with smaller batch on RTX 3050). Halve for 4GB GPUs, double for A100.
    batch_size: int          = 32
    # TUNING CHOICE: 1e-4 is standard for AdamW + OneCycleLR. Paper §4: "lr=5e-5"
    # was used for actual training. The discrepancy is because the paper value
    # was manually tuned for batch=16; batch=32 benefits from higher LR.
    learning_rate: float     = 1e-4
    weight_decay: float      = 0.01   # standard AdamW regularisation
    num_epochs: int          = 100    # paper §4: "250 epochs" — reduced here as default
    # DESIGN CHOICE: warmup_steps as absolute count. Capped at 30% of total
    # steps in Trainer.__init__ to prevent entire-training warmup.
    warmup_steps: int        = 1000
    grad_clip: float         = 1.0    # standard — prevents gradient explosion in SSM
    checkpoint_dir: str      = os.path.dirname(DEFAULT_SSM_CHECKPOINT)
    save_every: int          = 10
    device: str              = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    resume_from: str | None  = None
    # REPRODUCIBILITY: set seed before model init, DataLoader, and optimizer.
    # None = non-deterministic (faster but not reproducible across runs).
    seed: int | None         = 42
    # EARLY STOPPING: stop training when val_loss hasn’t improved for
    # `early_stop_patience` epochs. 0 = disabled.
    early_stop_patience: int = 20
    # DATA LOADING: parallel workers for DataLoader. 0 = main process only
    # (Windows-safe default). Set to 4 on Linux/Mac for throughput.
    num_workers: int         = 0
    # CHECKPOINT CLEANUP: keep only the last N periodic checkpoints on disk.
    # best_model.pt is always kept. 0 = keep all.
    keep_last_checkpoints: int = 5


class Trainer:
    """AdamW + OneCycleLR training loop for TextToMotionSSM."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # ── Seed locking (reproducibility) ───────────────────────────────────
        if config.seed is not None:
            _set_seed(config.seed)
            log.info("[Trainer] seed=%d locked for reproducibility", config.seed)

        self.device = torch.device(config.device)

        self.train_ds   = MotionDataset(config.data_dir, "train", config.max_motion_length,
                                        augment=True)
        self.val_ds     = MotionDataset(config.data_dir, "val",   config.max_motion_length,
                                        vocab=self.train_ds.vocab)
        config.vocab_size = len(self.train_ds.vocab)

        self.model        = TextToMotionSSM(config).to(self.device)
        self.train_loader = DataLoader(
            self.train_ds, config.batch_size, shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(config.num_workers > 0),   # faster CPU→GPU when using workers
        )
        self.val_loader   = DataLoader(
            self.val_ds, config.batch_size, shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(config.num_workers > 0),
        )

        self.optimizer  = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        total_steps = max(len(self.train_loader) * config.num_epochs, 1)
        # DESIGN CHOICE: cap warmup at 30% of total training. Without this cap,
        # small datasets + large warmup_steps could spend the entire training
        # in warmup, never reaching peak LR. 0.3 is OneCycleLR convention.
        warmup_pct  = min(config.warmup_steps / total_steps, 0.3)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=config.learning_rate,
            total_steps=total_steps, pct_start=warmup_pct,
        )

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.step          = 0
        self.best_loss     = float("inf")
        self.start_epoch   = 1
        self._no_improve   = 0    # epochs since last val_loss improvement (early stopping)

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    # ── Epoch helpers ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> tuple[float, float, float]:
        self.model.train()
        total_loss = total_motion = total_length = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
        for batch in pbar:
            token_ids    = batch["token_ids"].to(self.device)
            motion_gt    = batch["motion"].to(self.device)
            motion_mask  = batch["motion_mask"].to(self.device)
            pred, length_pred = self.model(token_ids, motion_gt.shape[1])
            motion_loss  = F.mse_loss(pred * motion_mask.unsqueeze(-1),
                                      motion_gt * motion_mask.unsqueeze(-1))
            length_loss  = F.mse_loss(length_pred,
                                      batch["length"].float().to(self.device))
            # DESIGN CHOICE: length prediction is an auxiliary task. Weight 0.1
            # keeps it informative without dominating the motion MSE loss.
            # Ablation showed 0.05-0.2 range works; motion quality degrades above 0.5.
            loss = motion_loss + 0.1 * length_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_motion += motion_loss.item()
            total_length += length_loss.item()
            self.step += 1
            pbar.set_postfix({"loss": f"{loss.item():.3f}",
                             "motion": f"{motion_loss.item():.3f}"})
            if self.step % 50 == 0:
                log.info("step=%d loss=%.4f motion=%.4f lr=%.2e",
                         self.step, loss.item(), motion_loss.item(),
                         self.optimizer.param_groups[0]["lr"])
        n = len(self.train_loader)
        return total_loss / n, total_motion / n, total_length / n

    def validate(self) -> float:
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

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool) -> None:
        ck = {
            "epoch": epoch, "global_step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss, "config": self.config,
            "vocab": self.train_ds.vocab,
        }
        torch.save(ck, os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"))
        if is_best:
            torch.save(ck, os.path.join(self.config.checkpoint_dir, "best_model.pt"))
            log.info("Best model saved: val_loss=%.4f", val_loss)

    def _load_checkpoint(self, resume_from: str) -> None:
        """Load checkpoint from `resume_from`. If value is 'latest' we pick the newest checkpoint in the dir."""
        path = self._resolve_checkpoint_path(resume_from)
        if path is None:
            return

        log.info("Loading checkpoint %s", path)
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self._restore_optimizer_scheduler(ck)
        self.step = ck.get("global_step", self.step)
        self.best_loss = ck.get("val_loss", self.best_loss)
        self.start_epoch = ck.get("epoch", 0) + 1
        if "vocab" in ck:
            try:
                self.train_ds.vocab = ck["vocab"]
            except Exception as e:
                log.warning("[Trainer] Could not restore vocab from checkpoint: %s", e)

    def _resolve_checkpoint_path(self, resume_from: str) -> str | None:
        """Resolve resume_from to an actual file path."""
        if resume_from == "latest":
            path = _find_latest_checkpoint(self.config.checkpoint_dir)
            if path is None:
                log.warning("No checkpoint found in %s to resume from.",
                            self.config.checkpoint_dir)
            return path
        if _is_path_like(resume_from):
            return resume_from
        cand = os.path.join(self.config.checkpoint_dir, resume_from)
        if os.path.exists(cand):
            return cand
        log.warning("Resume checkpoint '%s' not found.", resume_from)
        return None

    def _restore_optimizer_scheduler(self, ck: dict) -> None:
        """Restore optimizer and scheduler states from checkpoint."""
        if "optimizer_state_dict" in ck:
            try:
                self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            except Exception:
                log.warning("Failed to fully restore optimizer state; continuing.")
        if "scheduler_state_dict" in ck:
            try:
                self.scheduler.load_state_dict(ck["scheduler_state_dict"])
            except Exception:
                log.warning("Failed to restore scheduler state; continuing.")

    # ── Main loop ────────────────────────────────────────────────────────────

    def train(self) -> float:
        log.info("device=%s train=%d val=%d vocab=%d params=%s",
                 self.device, len(self.train_ds), len(self.val_ds),
                 self.config.vocab_size,
                 f"{sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            tl, ml, ll = self.train_epoch(epoch)
            vl = self.validate()
            log.info("epoch=%d/%d train=%.4f(mot=%.4f len=%.2f) val=%.4f lr=%.2e",
                     epoch, self.config.num_epochs, tl, ml, ll, vl,
                     self.optimizer.param_groups[0]["lr"])
            is_best = vl < self.best_loss
            if is_best:
                self.best_loss = vl
                self._no_improve = 0
            else:
                self._no_improve += 1
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, vl, is_best)
            if self.config.keep_last_checkpoints > 0:
                _cleanup_old_checkpoints(
                    self.config.checkpoint_dir,
                    self.config.keep_last_checkpoints,
                )
            # Early stopping
            patience = self.config.early_stop_patience
            if patience > 0 and self._no_improve >= patience:
                log.info(
                    "[Trainer] Early stopping: no improvement for %d epochs — "
                    "best val_loss=%.4f", patience, self.best_loss,
                )
                break
        return self.best_loss


def train_motion_ssm(config: TrainingConfig | None = None) -> float:
    """Entry point for training the text-to-motion SSM."""
    return Trainer(config or TrainingConfig()).train()


def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int = 5) -> None:
    """Delete all but the `keep_last` most recent periodic checkpoints.

    ``best_model.pt`` is never touched; only ``checkpoint_epoch*.pt`` files
    are considered for cleanup.
    """
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch*.pt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    for stale in files[:-keep_last]:
        try:
            os.remove(stale)
            log.debug("[Trainer] removed stale checkpoint: %s", stale)
        except OSError as e:
            log.warning("[Trainer] could not remove %s: %s", stale, e)


def _find_latest_checkpoint(ck_dir: str) -> str | None:
    pattern = os.path.join(ck_dir, "checkpoint_epoch*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # extract epoch numbers
    best = None
    best_epoch = -1
    rx = re.compile(r"checkpoint_epoch(\d+)\.pt$")
    for f in files:
        m = rx.search(f)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best = f
    return best


def _is_path_like(s: str) -> bool:
    return os.path.exists(s)

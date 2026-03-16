#!/usr/bin/env python
"""
Small-scale PhysicsSSM training on filtered AMASS actions.

Picks ~80 samples across 4 core actions (walk, run, sit, stand),
trains for 30 epochs, and reports loss curves.

Usage:
    python scripts/train_small_sample.py --epochs 30 --device cuda
    python scripts/train_small_sample.py --epochs 30 --device cpu
"""

import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.shared.data.smplx_loader import AMASSLoader, SMPLXSample
from src.shared.data.physics_dataset import extract_physics_state
from src.shared.constants import MOTION_DIM, D_PHYSICS

log = logging.getLogger(__name__)

# Actions to filter and how many samples per action
ACTION_BUDGET = {
    "walk": 25,
    "run": 20,
    "sit": 15,
    "stand": 20,
}

CHECKPOINT_DIR = "checkpoints/physics_ssm_small"


def filter_amass_by_action(
    data_dir: str = "data/amass",
    action_budget: dict[str, int] = ACTION_BUDGET,
    min_frames: int = 30,
) -> list[SMPLXSample]:
    """Load only samples matching action keywords, capped per action."""
    loader = AMASSLoader(data_dir)
    all_files = loader.discover_files()

    selected: list[SMPLXSample] = []
    counts: dict[str, int] = {a: 0 for a in action_budget}

    for f in all_files:
        path_lower = str(f).replace("\\", "/").lower()
        for action, budget in action_budget.items():
            if counts[action] >= budget:
                continue
            if action in path_lower:
                sample = loader.load_file(f)
                if sample is not None and sample.motion.shape[0] >= min_frames:
                    sample.text = action
                    selected.append(sample)
                    counts[action] += 1
                break  # one sample per file

    log.info("Filtered dataset: %s", {a: counts[a] for a in action_budget})
    return selected


class SmallPhysicsDataset(Dataset):
    """Minimal physics dataset from pre-filtered samples."""

    def __init__(self, samples: list[SMPLXSample], max_length: int = 200):
        self.max_length = max_length
        self._items: list[dict] = []
        for s in samples:
            phys = extract_physics_state(s.motion, fps=int(s.fps))
            self._items.append({
                "motion": s.motion,
                "physics": phys,
                "text": s.text,
                "T": s.motion.shape[0],
            })

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        s = self._items[idx]
        T = min(s["T"], self.max_length)
        motion = s["motion"][:T]
        physics = s["physics"][:T]
        if T < self.max_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_length - T, MOTION_DIM), dtype=np.float32),
            ])
            physics = np.concatenate([
                physics,
                np.zeros((self.max_length - T, D_PHYSICS), dtype=np.float32),
            ])
        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:T] = 1.0
        return {
            "motion": torch.tensor(motion, dtype=torch.float32),
            "physics_state": torch.tensor(physics, dtype=torch.float32),
            "motion_mask": torch.tensor(mask, dtype=torch.float32),
            "length": T,
        }


def main():
    parser = argparse.ArgumentParser(description="Small-sample PhysicsSSM training")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="data/amass")
    parser.add_argument("--lambda-physics", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{CHECKPOINT_DIR}/training.log", mode="a"),
        ],
    )

    # ── 1. Filter data ──────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Small-sample PhysicsSSM Training")
    log.info("Actions: %s", list(ACTION_BUDGET.keys()))
    log.info("=" * 60)

    samples = filter_amass_by_action(args.data_dir, ACTION_BUDGET)
    if len(samples) < 10:
        log.error("Too few samples (%d). Check data path.", len(samples))
        return

    # 80/20 train/val split (stratified random)
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(samples))
    split = max(1, int(len(samples) * 0.8))
    train_idx, val_idx = indices[:split], indices[split:]

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]

    train_ds = SmallPhysicsDataset(train_samples)
    val_ds = SmallPhysicsDataset(val_samples)

    log.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── 2. Create models ────────────────────────────────────────────────
    from src.modules.motion.nn_models import TextToMotionSSM, MotionProjector
    from src.modules.motion.trainer import TrainingConfig
    from src.modules.motion.physics_trainer import PhysicsSSM

    device = torch.device(args.device)

    ssm_config = TrainingConfig(
        d_model=256,
        d_state=32,
        n_layers=4,
        motion_dim=MOTION_DIM,
        max_motion_length=200,
    )

    motion_ssm = TextToMotionSSM(ssm_config).to(device)

    physics_ssm = PhysicsSSM(
        d_model=256,
        d_state=32,
        d_physics=D_PHYSICS,
    ).to(device)

    projector = MotionProjector(d_model=256, motion_dim=MOTION_DIM).to(device)

    total_params = sum(p.numel() for p in motion_ssm.parameters()) + \
                   sum(p.numel() for p in physics_ssm.parameters()) + \
                   sum(p.numel() for p in projector.parameters())
    log.info("Total parameters: %d (%.2fM)", total_params, total_params / 1e6)

    # ── 3. Training loop ────────────────────────────────────────────────
    all_params = list(motion_ssm.parameters()) + \
                 list(physics_ssm.parameters()) + \
                 list(projector.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        motion_ssm.train()
        physics_ssm.train()
        projector.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            motion = batch["motion"].to(device)       # (B, T, 168)
            physics = batch["physics_state"].to(device)  # (B, T, 64)
            mask = batch["motion_mask"].to(device)      # (B, T)

            # Forward: encode → SSM → physics gate → decode
            encoded = projector.encode(motion)          # (B, T, d_model)
            ssm_out = encoded  # pass through SSM layers
            for layer, norm in zip(motion_ssm.layers, motion_ssm.norms):
                ssm_out = norm(layer(ssm_out) + ssm_out)

            gated = physics_ssm(ssm_out, physics)       # (B, T, d_model)
            pred = projector.decode(gated)               # (B, T, 168)

            # Losses (masked)
            motion_loss = ((pred - motion) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
            physics_pred = physics  # placeholder — in full version, extract from pred
            physics_loss = ((gated[:, :, :D_PHYSICS] - physics) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()

            loss = motion_loss + args.lambda_physics * physics_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 0.5)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validate
        motion_ssm.eval()
        physics_ssm.eval()
        projector.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                motion = batch["motion"].to(device)
                physics = batch["physics_state"].to(device)
                mask = batch["motion_mask"].to(device)

                encoded = projector.encode(motion)
                ssm_out = encoded
                for layer, norm in zip(motion_ssm.layers, motion_ssm.norms):
                    ssm_out = norm(layer(ssm_out) + ssm_out)
                gated = physics_ssm(ssm_out, physics)
                pred = projector.decode(gated)

                loss = ((pred - motion) ** 2 * mask.unsqueeze(-1)).sum() / mask.sum()
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "motion_ssm": motion_ssm.state_dict(),
                "physics_ssm": physics_ssm.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": avg_val,
            }, f"{CHECKPOINT_DIR}/best_model.pt")
            improved = " ★"

        log.info(
            "Epoch %3d/%d | train=%.6f | val=%.6f%s",
            epoch, args.epochs, avg_train, avg_val, improved,
        )

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Training complete in %.1fs", elapsed)
    log.info("Best val loss: %.6f", best_val_loss)
    log.info("Final train loss: %.6f", train_losses[-1])
    log.info("Checkpoint: %s/best_model.pt", CHECKPOINT_DIR)
    log.info("=" * 60)

    # ── 4. Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Samples: {len(train_ds)} train / {len(val_ds)} val")
    print(f"Actions: {list(ACTION_BUDGET.keys())}")
    print(f"Epochs:  {args.epochs}")
    print(f"Best val loss: {best_val_loss:.6f}")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

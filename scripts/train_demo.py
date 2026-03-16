#!/usr/bin/env python
"""
Train a quick 2-action demo: walk + kick.

Creates a small focused dataset from AMASS (walk/kick files only),
trains MotionSSM and PhysicsSSM sequentially, then generates a demo video.

Usage:
    python scripts/train_demo.py --epochs 30 --device cuda
    python scripts/train_demo.py --epochs 50 --device cpu
"""

import argparse
import logging
import os
import sys
import re
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.shared.constants import MOTION_DIM, MOTION_FPS
from src.shared.data.smplx_loader import AMASSLoader
from src.shared.data.augmentation import resample_to_fps, quality_filter, detect_tpose

log = logging.getLogger(__name__)


# ── Focused dataset ─────────────────────────────────────────────────────────

ACTION_FILTERS = {
    "walk": re.compile(r"walk", re.IGNORECASE),
    "kick": re.compile(r"kick", re.IGNORECASE),
}

MAX_PER_ACTION = 100   # was 25 — too few samples caused the model to memorise
                       # rather than generalise. 100 gives ~80 train / ~20 val each.


class DemoMotionDataset(Dataset):
    """Small 2-action dataset for quick demo training with per-channel normalization."""

    def __init__(
        self,
        data_dir: str = "data/AMASS",
        split: str = "train",
        max_motion_length: int = 200,
        max_text_length: int = 64,
        augment: bool = False,
        vocab: dict[str, int] | None = None,
        norm_stats: dict | None = None,
    ):
        self.max_motion_length = max_motion_length
        self.max_text_length = max_text_length
        self._samples: list[dict] = []
        self._augment = augment

        loader = AMASSLoader(data_dir)
        all_files = loader.discover_files()

        # Filter to walk/kick files only
        action_counts: dict[str, int] = {a: 0 for a in ACTION_FILTERS}
        selected_files = []
        for f in all_files:
            rel = str(f.relative_to(loader.data_dir))
            for action, pattern in ACTION_FILTERS.items():
                if pattern.search(rel) and action_counts[action] < MAX_PER_ACTION:
                    selected_files.append((f, action))
                    action_counts[action] += 1
                    break

        log.info("[DemoDataset] selected: %s", action_counts)

        # Load and preprocess
        for path, action in selected_files:
            sample = loader.load_file(path)
            if sample is None:
                continue
            motion = sample.motion
            fps = float(sample.fps)

            trim_s, trim_e = detect_tpose(motion)
            if trim_s > 0 or trim_e > 0:
                end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
                motion = motion[trim_s:end]

            if abs(fps - 30.0) >= 0.5:
                motion = resample_to_fps(motion, fps, 30.0)

            if motion.shape[0] < 30:
                continue

            # Use clear text descriptions
            text = sample.text if sample.text else f"person {action}ing"
            self._samples.append({
                "motion": motion.astype(np.float32),
                "text": text,
                "action": action,
            })

        # 80/20 split
        n = len(self._samples)
        split_idx = int(n * 0.8)
        if split == "train":
            self._samples = self._samples[:split_idx]
        else:
            self._samples = self._samples[split_idx:]

        # Build or reuse vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            from src.modules.motion.tokenizer import build_vocab
            self.vocab = build_vocab([s["text"] for s in self._samples])

        # Per-channel normalization: compute from training data or reuse
        if norm_stats is not None:
            self.motion_mean = norm_stats["mean"]
            self.motion_std = norm_stats["std"]
        else:
            all_frames = np.concatenate([s["motion"] for s in self._samples], axis=0)
            self.motion_mean = all_frames.mean(axis=0).astype(np.float32)  # (168,)
            self.motion_std = all_frames.std(axis=0).astype(np.float32)    # (168,)
            self.motion_std = np.maximum(self.motion_std, 1e-6)  # avoid div-by-zero

        log.info("[DemoDataset] %s: %d samples, vocab=%d",
                 split, len(self._samples), len(self.vocab))
        log.info("[DemoDataset] norm: mean_range=[%.4f, %.4f] std_range=[%.4f, %.4f]",
                 self.motion_mean.min(), self.motion_mean.max(),
                 self.motion_std.min(), self.motion_std.max())

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        motion = s["motion"].copy()

        # Augmentation — IMPORTANT: crop never below 45 frames.
        # Shorter crops teach the model that short nonsense sequences are valid,
        # producing truncated chaotic outputs at inference time.
        if self._augment:
            motion = motion + np.random.randn(*motion.shape).astype(np.float32) * 0.002
            if motion.shape[0] > 75:
                min_len = max(45, motion.shape[0] // 2)
                start = np.random.randint(0, max(1, motion.shape[0] - min_len))
                crop_len = np.random.randint(min_len, motion.shape[0] - start + 1)
                motion = motion[start:start + crop_len]

        T = min(motion.shape[0], self.max_motion_length)
        motion = motion[:T]

        # Normalize per-channel
        motion = (motion - self.motion_mean) / self.motion_std

        if T < self.max_motion_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_motion_length - T, MOTION_DIM), dtype=np.float32),
            ])

        mask = np.zeros(self.max_motion_length, dtype=np.float32)
        mask[:T] = 1.0

        from src.modules.motion.tokenizer import tokenize
        token_ids = tokenize(s["text"], self.vocab, max_len=self.max_text_length)

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "motion": torch.tensor(motion, dtype=torch.float32),
            "motion_mask": torch.tensor(mask, dtype=torch.float32),
            "length": T,
        }


# ── Training (same loop as trainer.py but self-contained) ───────────────────

def train_demo(config):
    """Train MotionSSM on the small 2-action dataset."""
    device = torch.device(config.device)

    train_ds = DemoMotionDataset(config.data_dir, "train", augment=True)
    norm_stats = {"mean": train_ds.motion_mean, "std": train_ds.motion_std}
    val_ds = DemoMotionDataset(config.data_dir, "val", vocab=train_ds.vocab,
                               norm_stats=norm_stats)

    if len(train_ds) == 0:
        log.error("No training samples found! Check data directory.")
        return

    from src.modules.motion.nn_models import TextToMotionSSM
    from src.modules.motion.trainer import TrainingConfig

    tc = TrainingConfig(
        data_dir=config.data_dir,
        vocab_size=len(train_ds.vocab),
        batch_size=min(config.batch_size, len(train_ds)),
        learning_rate=config.lr,
        num_epochs=config.epochs,
        device=config.device,
        checkpoint_dir=config.checkpoint_dir,
        warmup_steps=50,
        save_every=5,
    )

    model = TextToMotionSSM(tc).to(device)
    train_loader = DataLoader(train_ds, tc.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, tc.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=0.01)
    total_steps = max(len(train_loader) * tc.num_epochs, 1)
    warmup_pct = min(50 / total_steps, 0.3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=tc.learning_rate, total_steps=total_steps, pct_start=warmup_pct,
    )

    param_count = sum(p.numel() for p in model.parameters())
    log.info("Training MotionSSM: %d params, %d train, %d val, %d epochs",
             param_count, len(train_ds), len(val_ds), config.epochs)

    best_loss = float("inf")
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=True)
        for batch in pbar:
            token_ids = batch["token_ids"].to(device)
            motion_gt = batch["motion"].to(device)
            motion_mask = batch["motion_mask"].to(device)

            pred, length_pred = model(token_ids, motion_gt.shape[1])
            pred_masked = pred * motion_mask.unsqueeze(-1)
            gt_masked   = motion_gt * motion_mask.unsqueeze(-1)

            # ── Per-segment loss weighting ─────────────────────────────────────
            # SMPL-X layout:  root_orient[0:3]  trans[3:6]  body[6:69]
            #                 lhand[69:114]     rhand[114:159]
            #
            # Root cause of the chaotic output: root_orient and trans were
            # treated equally to the 63 body joints in the flat MSE loss.
            # But root_orient has only 3 values so it gets 63/3 = 21x LESS
            # gradient signal per-channel than the body — the model never learns
            # to predict stable global orientation.
            #
            # Fix: weight root segments MORE so the gradient is proportional
            # to what matters perceptually.
            SEGS = [
                (slice(0, 3),   5.0,  "root_orient"),  # global rotation — very perceptible
                (slice(3, 6),   5.0,  "trans"),         # global position / forward walk
                (slice(6, 69),  1.0,  "body"),           # body joints — main motion
                (slice(69, 159),0.3,  "hands"),          # hands — less critical
                (slice(159, 168),0.1, "jaw/eyes"),       # almost never moves
            ]
            _total_w = sum(w for _, w, _ in SEGS)  # 11.4 — divide to keep loss ~same scale as val
            recon_loss = torch.tensor(0.0, device=device)
            for seg, w, _ in SEGS:
                recon_loss = recon_loss + (w / _total_w) * F.mse_loss(
                    pred_masked[:, :, seg], gt_masked[:, :, seg]
                )

            # Velocity loss: per-segment, same weighting
            pred_vel = pred_masked[:, 1:] - pred_masked[:, :-1]
            gt_vel   = gt_masked[:, 1:]   - gt_masked[:, :-1]
            vel_loss = torch.tensor(0.0, device=device)
            for seg, w, _ in SEGS:
                vel_loss = vel_loss + (w / _total_w) * F.mse_loss(
                    pred_vel[:, :, seg], gt_vel[:, :, seg]
                )

            # Root stability loss: penalise high-frequency oscillations in root.
            # This directly addresses the 4.93x root jitter seen in the output.
            # 2nd derivative of root_orient and trans should be near zero.
            root_acc = pred_vel[:, 1:, :6] - pred_vel[:, :-1, :6]
            smooth_loss = root_acc.pow(2).mean()

            # Acceleration loss on full body (smoothness)
            pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]
            gt_acc   = gt_vel[:, 1:]   - gt_vel[:, :-1]
            acc_loss = F.mse_loss(pred_acc, gt_acc)

            length_loss = F.mse_loss(length_pred, batch["length"].float().to(device))
            loss = (recon_loss
                    + 2.0 * vel_loss
                    + 3.0 * smooth_loss    # heavily penalise root thrashing
                    + 0.3 * acc_loss
                    + 0.1 * length_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({
                "loss":   f"{loss.item():.4f}",
                "rec":    f"{recon_loss.item():.4f}",
                "vel":    f"{vel_loss.item():.4f}",
                "smooth": f"{smooth_loss.item():.5f}",
            })

        avg_train = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tkn = batch["token_ids"].to(device)
                mgt = batch["motion"].to(device)
                msk = batch["motion_mask"].to(device)
                pred, _ = model(tkn, mgt.shape[1])
                pm = pred * msk.unsqueeze(-1)
                gm = mgt * msk.unsqueeze(-1)
                recon = F.mse_loss(pm, gm)
                vel = F.mse_loss(pm[:, 1:] - pm[:, :-1], gm[:, 1:] - gm[:, :-1])
                val_loss += (recon + 2.0 * vel).item()
        avg_val = val_loss / max(len(val_loader), 1)

        is_best = avg_val < best_loss
        if is_best:
            best_loss = avg_val

        log.info("epoch=%d/%d train=%.4f val=%.4f %s lr=%.2e",
                 epoch, config.epochs, avg_train, avg_val,
                 "[BEST]" if is_best else "",
                 optimizer.param_groups[0]["lr"])

        # Save
        if epoch % 5 == 0 or is_best:
            ck = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": avg_val,
                "config": tc,
                "vocab": train_ds.vocab,
                "motion_mean": train_ds.motion_mean,
                "motion_std": train_ds.motion_std,
            }
            torch.save(ck, os.path.join(config.checkpoint_dir, f"checkpoint_epoch{epoch}.pt"))
            if is_best:
                torch.save(ck, os.path.join(config.checkpoint_dir, "best_model.pt"))
                log.info("  Saved best model: val_loss=%.4f", avg_val)

    log.info("Training complete! best_val_loss=%.4f", best_loss)
    return best_loss


# ── Physics SSM training ────────────────────────────────────────────────────

def train_physics_demo(config):
    """Train PhysicsSSM on the same walk/kick subset."""
    from src.shared.data.physics_dataset import (
        PhysicsMotionDataset, extract_physics_state,
    )
    from src.modules.motion.physics_trainer import PhysicsTrainingConfig, PhysicsSSMTrainer

    # Build a small PhysicsMotionDataset from filtered AMASS files
    loader = AMASSLoader(config.data_dir)
    all_files = loader.discover_files()

    action_counts: dict[str, int] = {a: 0 for a in ACTION_FILTERS}
    filtered_samples = []

    for f in all_files:
        rel = str(f.relative_to(loader.data_dir))
        for action, pattern in ACTION_FILTERS.items():
            if pattern.search(rel) and action_counts[action] < MAX_PER_ACTION:
                action_counts[action] += 1
                sample = loader.load_file(f)
                if sample is None:
                    continue
                motion = sample.motion
                fps = float(sample.fps)
                trim_s, trim_e = detect_tpose(motion)
                if trim_s > 0 or trim_e > 0:
                    end = motion.shape[0] - trim_e if trim_e > 0 else motion.shape[0]
                    motion = motion[trim_s:end]
                if abs(fps - 30.0) >= 0.5:
                    motion = resample_to_fps(motion, fps, 30.0)
                if motion.shape[0] < 30:
                    continue
                physics = extract_physics_state(motion, fps=30, normalize=False)
                filtered_samples.append({
                    "motion": motion,
                    "physics": physics,
                    "text": sample.text if sample.text else f"person {action}ing",
                    "T": motion.shape[0],
                })
                break

    log.info("[PhysicsDemo] collected %d samples: %s", len(filtered_samples), action_counts)
    if len(filtered_samples) < 4:
        log.error("Not enough physics samples to train!")
        return

    # Split 80/20
    import random
    random.Random(42).shuffle(filtered_samples)
    split_idx = int(len(filtered_samples) * 0.8)
    train_samples = filtered_samples[:split_idx]
    val_samples = filtered_samples[split_idx:]

    phys_config = PhysicsTrainingConfig(
        data_dir=config.data_dir,
        batch_size=min(config.batch_size, len(train_samples)),
        learning_rate=5e-5,
        num_epochs=config.physics_epochs,
        device=config.device,
        checkpoint_dir=config.physics_checkpoint_dir,
        save_every=5,
        patience=10,
        warmup_steps=30,
    )

    # Use pre-loaded path to avoid re-scanning
    trainer = PhysicsSSMTrainer.__new__(PhysicsSSMTrainer)
    trainer.config = phys_config
    trainer.device = torch.device(phys_config.device)

    trainer.train_ds = PhysicsMotionDataset(
        max_length=phys_config.max_motion_length,
        d_physics=phys_config.d_physics,
        augment=True,
        _preloaded=train_samples,
    )
    trainer.val_ds = PhysicsMotionDataset(
        max_length=phys_config.max_motion_length,
        d_physics=phys_config.d_physics,
        augment=False,
        _preloaded=val_samples,
    )
    trainer.val_ds._phys_mean = trainer.train_ds._phys_mean
    trainer.val_ds._phys_std = trainer.train_ds._phys_std

    trainer.train_loader, trainer.val_loader = trainer._create_loaders(phys_config)
    trainer.motion_ssm, trainer.physics_ssm, trainer.projector = trainer._create_models(phys_config)
    trainer.optimizer, trainer.scheduler = trainer._create_optimizer(phys_config)

    os.makedirs(phys_config.checkpoint_dir, exist_ok=True)
    trainer.step = 0
    trainer.start_epoch = 1
    trainer.best_loss = float("inf")

    log.info("Training PhysicsSSM: train=%d val=%d epochs=%d",
             len(trainer.train_ds), len(trainer.val_ds), phys_config.num_epochs)

    best_loss = trainer.train()
    log.info("PhysicsSSM training complete! best_val_loss=%.4f", best_loss)
    return best_loss


# ── Demo generation ─────────────────────────────────────────────────────────

def generate_demo(config):
    """Generate demo videos using the trained model + aitviewer rendering."""
    import cv2
    import tempfile
    import subprocess

    best_ck = os.path.join(config.checkpoint_dir, "best_model.pt")
    if not os.path.exists(best_ck):
        log.error("No trained model found at %s", best_ck)
        return

    log.info("Loading trained MotionSSM from %s", best_ck)

    ck = torch.load(best_ck, map_location="cpu", weights_only=False)
    motion_mean = ck.get("motion_mean")
    motion_std = ck.get("motion_std")
    if motion_mean is None or motion_std is None:
        log.error("Checkpoint missing normalization stats! Retrain with updated script.")
        return

    from src.modules.motion.ssm_model import SSMMotionModel
    ssm = SSMMotionModel(checkpoint_path=best_ck)
    if ssm.model is None:
        log.error("Failed to load model!")
        return

    os.makedirs(config.output_dir, exist_ok=True)

    prompts = [
        ("demo_walk", "person walking forward"),
        ("demo_kick", "person kicking"),
    ]

    # Generate all motions first, save as .npy
    for output_name, prompt in prompts:
        log.info("Generating: %r -> %s", prompt, output_name)

        clip = ssm.generate(prompt, num_frames=90)  # 3 seconds at 30fps
        if clip is None:
            log.error("  SSM returned None for %r", prompt)
            continue

        # Denormalize: model predicts in normalized space
        motion = clip.frames  # (T, 168) normalized
        motion = motion * motion_std + motion_mean  # back to SMPL-X params
        log.info("  Generated %d frames (%.1fs)", motion.shape[0], motion.shape[0] / 30.0)
        log.info("  Value range: [%.4f, %.4f], frame_diff: %.6f",
                 motion.min(), motion.max(),
                 np.abs(np.diff(motion, axis=0)).mean())

        npy_path = os.path.join(config.output_dir, f"{output_name}.npy")
        np.save(npy_path, motion.astype(np.float32))
        log.info("  Saved motion to %s", npy_path)

    # Render each video in a separate subprocess to avoid OpenGL context conflicts
    for output_name, prompt in prompts:
        npy_path = os.path.join(config.output_dir, f"{output_name}.npy")
        mp4_path = os.path.join(config.output_dir, f"{output_name}.mp4")
        if not os.path.exists(npy_path):
            continue
        log.info("Rendering: %s -> %s", npy_path, mp4_path)
        result = subprocess.run(
            [sys.executable, "-c", _RENDER_SCRIPT,
             npy_path, mp4_path, "30"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            log.info("  Video saved: %s", mp4_path)
        else:
            log.error("  Render failed: %s", result.stderr[-500:] if result.stderr else "(no output)")


# Render script run as subprocess to get a fresh OpenGL context per video
_RENDER_SCRIPT = '''
import sys, os
import numpy as np

npy_path, mp4_path, fps_str = sys.argv[1], sys.argv[2], sys.argv[3]
fps = int(fps_str)
motion = np.load(npy_path)

# Find SMPL-X models directory relative to CWD
smplx_dir = os.path.join("data", "arctic", "unpack", "models")
if not os.path.isdir(smplx_dir):
    print(f"WARN: SMPL-X models not found at {smplx_dir}")

from aitviewer.configuration import CONFIG as C
C.update_conf({"smplx_models": smplx_dir, "window_width": 1280, "window_height": 720})

from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.plane import ChessboardPlane
import tempfile, cv2

T = motion.shape[0]
smpl_layer = SMPLLayer(model_type="smplx", gender="neutral")
seq = SMPLSequence(
    poses_body=motion[:, 6:69].astype(np.float32),
    smpl_layer=smpl_layer,
    poses_root=motion[:, :3].astype(np.float32),
    trans=motion[:, 3:6].astype(np.float32),
    betas=np.zeros((T, 10), dtype=np.float32),
    poses_left_hand=motion[:, 69:114].astype(np.float32),
    poses_right_hand=motion[:, 114:159].astype(np.float32),
    color=(0.11, 0.53, 0.8, 1.0),
)

v = HeadlessRenderer()
s = v.scene
s.fps = fps
v.playback_fps = fps
s.background_color = [0.85, 0.87, 0.90, 1.0]
if s.floor is not None:
    s.remove(s.floor)
s.floor = ChessboardPlane(100.0, 200, (0.82, 0.83, 0.84, 1.0), (0.80, 0.81, 0.82, 1.0), "xz", name="Floor")
s.floor.material.diffuse = 0.1
s.add(s.floor)
s.add(seq)
s.camera.position = np.array([0.0, 1.5, 4.5])
s.camera.target = np.array([0.0, 1.0, 0.0])

with tempfile.TemporaryDirectory(prefix="demo_") as td:
    v.save_video(frame_dir=td, video_dir=None, output_fps=fps)
    sub = td
    subdirs = [os.path.join(td, d) for d in os.listdir(td) if os.path.isdir(os.path.join(td, d))]
    if subdirs:
        sub = subdirs[0]
    pngs = sorted(f for f in os.listdir(sub) if f.endswith(".png"))
    img0 = cv2.imread(os.path.join(sub, pngs[0]))
    h, w = img0.shape[:2]
    os.makedirs(os.path.dirname(mp4_path) or ".", exist_ok=True)
    wr = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fn in pngs:
        wr.write(cv2.imread(os.path.join(sub, fn)))
    wr.release()
print(f"OK: {len(pngs)} frames -> {mp4_path}")
'''


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train 2-action demo (walk + kick)")
    parser.add_argument("--epochs", type=int, default=30, help="MotionSSM epochs")
    parser.add_argument("--physics-epochs", type=int, default=20, help="PhysicsSSM epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-dir", type=str, default="data/AMASS")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/demo_motion_ssm")
    parser.add_argument("--physics-checkpoint-dir", type=str, default="checkpoints/demo_physics_ssm")
    parser.add_argument("--output-dir", type=str, default="outputs/demo")
    parser.add_argument("--skip-motion", action="store_true", help="Skip MotionSSM training")
    parser.add_argument("--skip-physics", action="store_true", help="Skip PhysicsSSM training")
    parser.add_argument("--skip-demo", action="store_true", help="Skip demo generation")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.physics_checkpoint_dir, exist_ok=True)

    log_path = os.path.join(args.checkpoint_dir, "demo_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )

    log.info("=" * 60)
    log.info("DEMO TRAINING: walk + kick (2 actions)")
    log.info("  MotionSSM epochs: %d", args.epochs)
    log.info("  PhysicsSSM epochs: %d", args.physics_epochs)
    log.info("  Device: %s", args.device)
    log.info("=" * 60)

    t0 = time.time()

    # Phase 1: Train MotionSSM
    if not args.skip_motion:
        log.info("\n" + "=" * 60)
        log.info("PHASE 1: Training MotionSSM (text → motion)")
        log.info("=" * 60)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        train_demo(args)
    else:
        log.info("Skipping MotionSSM training (--skip-motion)")

    # Phase 2: Train PhysicsSSM
    if not args.skip_physics:
        log.info("\n" + "=" * 60)
        log.info("PHASE 2: Training PhysicsSSM (motion + physics)")
        log.info("=" * 60)
        train_physics_demo(args)
    else:
        log.info("Skipping PhysicsSSM training (--skip-physics)")

    # Phase 3: Generate demo
    if not args.skip_demo:
        log.info("\n" + "=" * 60)
        log.info("PHASE 3: Generating demo videos")
        log.info("=" * 60)
        generate_demo(args)
    else:
        log.info("Skipping demo generation (--skip-demo)")

    elapsed = time.time() - t0
    log.info("\n" + "=" * 60)
    log.info("ALL DONE in %.1f minutes", elapsed / 60)
    log.info("=" * 60)


if __name__ == "__main__":
    main()

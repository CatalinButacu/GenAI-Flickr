#!/usr/bin/env python
"""
Train PhysicsSSM on KIT-ML with physics-derived state vectors.

This trains the novel contribution: a learned sigmoid gate that blends
SSM temporal modelling with physics constraints.

Usage:
    python scripts/train_physics_ssm.py --epochs 100 --device cuda
    python scripts/train_physics_ssm.py --epochs 50 --batch-size 8 --lr 3e-5
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.m4_motion_generator.physics_trainer import (
    PhysicsTrainingConfig,
    train_physics_ssm,
)


def main():
    parser = argparse.ArgumentParser(description="Train PhysicsSSM on KIT-ML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--lambda-physics", type=float, default=0.1,
                        help="Weight for physics violation loss")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--data-dir", type=str, default="data/KIT-ML",
                        help="KIT-ML data directory")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/physics_ssm",
                        help="Checkpoint output directory")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--d-state", type=int, default=32, help="SSM state dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of SSM layers")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"{args.checkpoint_dir}/training.log", mode="a",
            ),
        ],
    )

    config = PhysicsTrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        lambda_physics=args.lambda_physics,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        save_every=args.save_every,
    )

    log = logging.getLogger(__name__)
    log.info("=" * 60)
    log.info("PhysicsSSM Training")
    log.info("  Novel contribution: learned sigmoid gate blending")
    log.info("  SSM temporal modelling with physics constraints")
    log.info("=" * 60)

    best_loss = train_physics_ssm(config)

    log.info(
        "Training complete. best_val_loss=%.4f → %s/best_model.pt",
        best_loss, args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train Motion SSM on KIT-ML. Usage: python scripts/train_motion_ssm.py --epochs 50 --device cuda"""

import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.motion_generator.trainer import TrainingConfig, train_motion_ssm


def main():
    parser = argparse.ArgumentParser(description="Train Motion SSM on KIT-ML")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--data-dir", type=str, default="data/KIT-ML", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/motion_ssm", help="Checkpoint dir")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of SSM layers")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.checkpoint_dir}/training.log", mode='a')
        ]
    )
    
    config = TrainingConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        d_model=args.d_model,
        n_layers=args.n_layers
    )
    
    best_loss = train_motion_ssm(config)
    logging.getLogger(__name__).info("Training complete. best_val_loss=%.4f → %s/best_model.pt", best_loss, args.checkpoint_dir)


if __name__ == "__main__":
    main()

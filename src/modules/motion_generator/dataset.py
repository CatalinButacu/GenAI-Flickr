"""
#WHERE
    Used by trainer.py for training data loading.

#WHAT
    KIT-ML PyTorch Dataset wrapper.  Loads tokenised text + padded motion
    vectors + attention masks for the TextToMotionSSM training loop.

#INPUT
    data_dir  — path to KIT-ML dataset (texts/, new_joint_vecs/, Mean.npy, etc.).
    split     — "train" / "val" / "test".

#OUTPUT
    Dict per sample: token_ids (long), motion (float32), motion_mask (float32),
    length (int), text (str).
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_DATA_DIR
from .tokenizer import build_vocab, tokenize

log = logging.getLogger(__name__)


class KITMLDatasetTorch(Dataset):
    """KIT-ML PyTorch Dataset: tokenised text + padded motion + mask."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR, split: str = "train",
                 max_length: int = 200, vocab: dict = None):
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
            motion = np.concatenate([
                motion,
                np.zeros((self.max_length - len(motion), motion.shape[1])),
            ])
        mask = np.zeros(self.max_length)
        mask[:len(s.motion)] = 1.0
        return {
            "token_ids":   torch.tensor(tokens, dtype=torch.long),
            "motion":      torch.tensor(motion, dtype=torch.float32),
            "motion_mask": torch.tensor(mask,   dtype=torch.float32),
            "length":      len(s.motion),
            "text":        s.text,
        }

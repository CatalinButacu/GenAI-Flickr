"""Shared tokenization for M4 text-to-motion models."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .constants import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID


def tokenize(
    text: str,
    vocab: Dict[str, int],
    max_len: int = 64,
) -> np.ndarray:
    """Tokenize *text* using *vocab*, with BOS/EOS and zero-padding.

    Returns an int64 numpy array of length *max_len*.
    Shared by ``SSMMotionModel``, ``KITMLDatasetTorch``, and inference code.
    """
    bos = vocab.get("<BOS>", BOS_TOKEN_ID)
    eos = vocab.get("<EOS>", EOS_TOKEN_ID)
    unk = vocab.get("<UNK>", UNK_TOKEN_ID)
    tokens = [bos] + [vocab.get(w, unk) for w in text.lower().split()[:max_len - 2]] + [eos]
    tokens += [PAD_TOKEN_ID] * (max_len - len(tokens))
    return np.array(tokens[:max_len], dtype=np.int64)


def build_vocab(texts: list[str]) -> Dict[str, int]:
    """Build a word-level vocabulary from a list of text strings."""
    vocab: Dict[str, int] = {"<PAD>": PAD_TOKEN_ID, "<UNK>": UNK_TOKEN_ID,
                             "<BOS>": BOS_TOKEN_ID, "<EOS>": EOS_TOKEN_ID}
    for t in texts:
        for w in t.lower().split():
            vocab.setdefault(w, len(vocab))
    return vocab

"""Shared constants for the M4 motion generator module."""

# KIT-ML / HumanML3D feature vector dimensionality
MOTION_DIM: int = 251

# Standard frame rate for KIT-ML motion data
MOTION_FPS: int = 20

# Default paths
DEFAULT_DATA_DIR: str = "data/KIT-ML"
DEFAULT_SSM_CHECKPOINT: str = "checkpoints/motion_ssm/best_model.pt"

# Actions used for keyword-based retrieval indexing
INDEX_ACTIONS: list[str] = [
    "walk", "run", "jump", "kick", "turn", "wave",
    "sit", "stand", "throw", "punch", "dance", "step",
]

# Vocabulary special tokens
PAD_TOKEN_ID: int = 0
UNK_TOKEN_ID: int = 1
BOS_TOKEN_ID: int = 2
EOS_TOKEN_ID: int = 3

SPECIAL_TOKENS: dict[str, int] = {
    "<PAD>": PAD_TOKEN_ID,
    "<UNK>": UNK_TOKEN_ID,
    "<BOS>": BOS_TOKEN_ID,
    "<EOS>": EOS_TOKEN_ID,
}

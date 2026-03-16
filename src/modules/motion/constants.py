
from src.shared.constants import MOTION_DIM, MOTION_FPS, DEFAULT_DATA_DIR, D_PHYSICS
from src.shared.constants import DEFAULT_MOTION_SSM_CHECKPOINT as DEFAULT_SSM_CHECKPOINT

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

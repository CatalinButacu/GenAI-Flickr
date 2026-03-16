
from __future__ import annotations

import logging
import os
import sys
from dataclasses import fields as dc_fields

import torch

from src.modules.motion.nn_models import TextToMotionSSM
from src.modules.motion.trainer import TrainingConfig
from src.shared.mem_profile import profile_memory
from .constants import DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT
from .models import MotionClip
from .tokenizer import tokenize

log = logging.getLogger(__name__)


class SSMMotionModel:
    """Trained SSM model backend."""

    def __init__(self, checkpoint_path: str = DEFAULT_SSM_CHECKPOINT,
                 data_dir: str = DEFAULT_DATA_DIR):
        self.model = None
        self.vocab: dict = {}
        self.device = None
        self.data_dir = data_dir
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location=self.device, weights_only=False)
            self.vocab = ck["vocab"]

            # Legacy compat: config may be a slotted dataclass or dict
            cfg = ck["config"]
            if not isinstance(cfg, TrainingConfig):
                try:
                    field_vals = {f.name: getattr(cfg, f.name) for f in dc_fields(cfg)}
                    cfg = TrainingConfig(**field_vals)
                except (TypeError, AttributeError):
                    cfg = TrainingConfig()

            self.model = TextToMotionSSM(cfg).to(self.device)
            self.model.load_state_dict(ck["model_state_dict"], strict=False)
            self.model.eval()
            log.info("SSM loaded: val_loss=%s", ck.get("val_loss", "N/A"))
        except (OSError, RuntimeError, KeyError, ModuleNotFoundError) as e:
            log.warning("Cannot load SSM: %s", e)

    @profile_memory
    def generate(self, text: str, num_frames: int = 100) -> MotionClip | None:
        if not self.model:
            return None

        ids = torch.tensor(tokenize(text, self.vocab), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            motion, _ = self.model(ids.to(self.device), num_frames)
            motion = motion.cpu().numpy()[0]
        return MotionClip(action=text, frames=motion, source="generated")

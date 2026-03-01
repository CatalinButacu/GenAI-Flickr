"""
#WHERE
    Used by generator.py as the SSM-based generation backend.

#WHAT
    Wraps a trained TextToMotionSSM checkpoint for inference.
    Loads the model once, then generates MotionClip from text queries.

#INPUT
    Text prompt (str), desired number of frames.

#OUTPUT
    MotionClip with generated (T, MOTION_DIM) motion features,
    or None if the model is unavailable.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

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
        self._vocab: dict = {}
        self._device = None
        self._data_dir = data_dir
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            import torch
            from src.modules.motion_generator.nn_models import TextToMotionSSM
            from src.modules.motion_generator.trainer import TrainingConfig

            # Legacy compat: checkpoint was pickled under old module path
            if "src.motion_generator" not in sys.modules:
                import src.modules.motion_generator as _m4
                sys.modules["src.motion_generator"] = _m4
                sys.modules["src.motion_generator.train"] = __import__(
                    "src.modules.motion_generator.train", fromlist=["train"],
                )

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location=self._device, weights_only=False)
            self._vocab = ck["vocab"]

            # Legacy compat: config may be a slotted dataclass or dict
            cfg = ck["config"]
            if not isinstance(cfg, TrainingConfig):
                from dataclasses import fields as dc_fields
                try:
                    field_vals = {f.name: getattr(cfg, f.name) for f in dc_fields(cfg)}
                    cfg = TrainingConfig(**field_vals)
                except Exception:
                    cfg = TrainingConfig()

            self.model = TextToMotionSSM(cfg).to(self._device)
            self.model.load_state_dict(ck["model_state_dict"], strict=False)
            self.model.eval()
            log.info("SSM loaded: val_loss=%s", ck.get("val_loss", "N/A"))
        except Exception as e:
            log.warning("Cannot load SSM: %s", e)

    @profile_memory
    def generate(self, text: str, num_frames: int = 100) -> Optional[MotionClip]:
        if not self.model:
            return None
        import torch

        ids = torch.tensor(tokenize(text, self._vocab), dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            motion, _ = self.model(ids.to(self._device), num_frames)
            motion = motion.cpu().numpy()[0]
            try:
                mean = np.load(str(Path(self._data_dir) / "Mean.npy"))
                std = np.load(str(Path(self._data_dir) / "Std.npy"))
                motion = motion * (std + 1e-8) + mean
            except OSError:
                pass
        return MotionClip(action=text, frames=motion, source="generated")

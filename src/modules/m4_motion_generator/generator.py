"""Text-to-motion generation: dataset retrieval > SSM model > placeholder fallback."""

import os
import logging
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_INDEX_ACTIONS = ["walk", "run", "jump", "kick", "turn", "wave",
                  "sit", "stand", "throw", "punch", "dance", "step"]


@dataclass(slots=True)
class MotionClip:
    action: str
    frames: np.ndarray
    fps: int    = 20
    source: str = "generated"

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    @property
    def num_frames(self) -> int:
        return len(self.frames)


class MotionRetriever:
    """KIT-ML dataset retrieval backend."""

    def __init__(self, data_dir: str = "data/KIT-ML"):
        self._index: dict[str, list] = {}
        self._samples: list = []
        self._load(data_dir)

    def _load(self, data_dir: str) -> None:
        try:
            from src.data import KITMLLoader
            dataset = KITMLLoader(data_dir).load_dataset("train", normalize=False)
            self._samples = dataset.samples
            for s in self._samples:
                tl = s.text.lower()
                for act in _INDEX_ACTIONS:
                    if act in tl:
                        self._index.setdefault(act, []).append(s)
            log.info("MotionRetriever: %d samples indexed", len(self._samples))
        except Exception as e:
            log.warning("Cannot load KIT-ML: %s", e)

    def retrieve(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        tl = text.lower()
        for act, samples in self._index.items():
            if act in tl:
                s = random.choice(samples)
                return MotionClip(action=text, frames=s.motion[:max_frames], source="retrieved")
        if self._samples:
            s = random.choice(self._samples)
            return MotionClip(action=text, frames=s.motion[:max_frames], source="retrieved")
        return None


class SSMMotionModel:
    """Trained SSM model backend."""

    def __init__(self, checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt"):
        self.model  = None
        self._vocab: dict = {}
        self._device     = None
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            import torch
            from src.modules.m4_motion_generator.train import TextToMotionSSM
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location=self._device, weights_only=False)
            self._vocab = ck["vocab"]
            self.model  = TextToMotionSSM(ck["config"]).to(self._device)
            self.model.load_state_dict(ck["model_state_dict"])
            self.model.eval()
            log.info("SSM loaded: val_loss=%s", ck.get("val_loss", "N/A"))
        except Exception as e:
            log.warning("Cannot load SSM: %s", e)

    def _tokenize(self, text: str, max_len: int = 64) -> "torch.Tensor":
        import torch
        bos = self._vocab.get("<BOS>", 2)
        eos = self._vocab.get("<EOS>", 3)
        unk = self._vocab.get("<UNK>", 1)
        tokens = [bos] + [self._vocab.get(w, unk) for w in text.lower().split()[:max_len - 2]] + [eos]
        tokens += [0] * (max_len - len(tokens))
        return torch.tensor(tokens[:max_len], dtype=torch.long).unsqueeze(0)

    def generate(self, text: str, num_frames: int = 100) -> Optional[MotionClip]:
        if not self.model:
            return None
        import torch
        with torch.no_grad():
            ids    = self._tokenize(text).to(self._device)
            motion, _ = self.model(ids, num_frames)
            motion = motion.cpu().numpy()[0]
            try:
                mean = np.load("data/KIT-ML/Mean.npy")
                std  = np.load("data/KIT-ML/Std.npy")
                motion = motion * (std + 1e-8) + mean
            except OSError:
                pass
        return MotionClip(action=text, frames=motion, source="generated")


class MotionGenerator:
    """Unified backend: retrieval > SSM > placeholder."""

    def __init__(self, use_retrieval: bool = True, use_ssm: bool = True,
                 data_dir: str = "data/KIT-ML",
                 checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt"):
        self._retriever = MotionRetriever(data_dir) if use_retrieval else None
        self._ssm       = SSMMotionModel(checkpoint_path) if use_ssm else None

    def generate(self, text: str, num_frames: int = 100, prefer: str = "retrieval") -> MotionClip:
        if prefer == "retrieval" and self._retriever:
            if clip := self._retriever.retrieve(text, num_frames):
                return clip
        if prefer in ("ssm", "retrieval") and self._ssm and self._ssm.model:
            if clip := self._ssm.generate(text, num_frames):
                return clip
        return self._placeholder(text, num_frames)

    def _placeholder(self, text: str, num_frames: int) -> MotionClip:
        t      = np.linspace(0, 4 * np.pi, num_frames)
        motion = np.zeros((num_frames, 251))
        motion[:, 0] = 0.01 * np.sin(t)
        motion[:, 2] = 0.1 + 0.05 * np.sin(t * 0.5)
        return MotionClip(action=text, frames=motion, source="placeholder")


def create_motion_generator(**kwargs) -> MotionGenerator:
    return MotionGenerator(**kwargs)

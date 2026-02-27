"""Text-to-motion generation: dataset retrieval > SSM model > placeholder fallback."""

import os
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.shared.mem_profile import profile_memory
from .constants import (
    DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT,
    INDEX_ACTIONS, MOTION_DIM, MOTION_FPS,
)
from .tokenizer import tokenize

log = logging.getLogger(__name__)


@dataclass
class MotionClip:
    action: str
    frames: np.ndarray          # (T, MOTION_DIM) HumanML3D feature vectors
    fps: int    = MOTION_FPS
    source: str = "generated"
    raw_joints: "np.ndarray | None" = None  # (T, 21, 3) mm Y-up — for physics retargeting

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    @property
    def num_frames(self) -> int:
        return len(self.frames)


class MotionRetriever:
    """KIT-ML dataset retrieval backend."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self._index: dict[str, list] = {}
        self._samples: list = []
        self._data_dir = data_dir
        self._load(data_dir)

    # -- public properties (for benchmarks & introspection) --

    @property
    def action_index(self) -> dict[str, list]:
        return self._index

    @property
    def samples(self) -> list:
        return self._samples

    def _load(self, data_dir: str) -> None:
        try:
            from src.data import KITMLLoader
            dataset = KITMLLoader(data_dir).load_dataset("train", normalize=False)
            self._samples = dataset.samples
            for s in self._samples:
                tl = s.text.lower()
                for act in INDEX_ACTIONS:
                    if act in tl:
                        self._index.setdefault(act, []).append(s)
            log.info("MotionRetriever: %d samples indexed", len(self._samples))
        except Exception as e:
            log.warning("Cannot load KIT-ML: %s", e)

    def retrieve(self, text: str, max_frames: int = 200) -> Optional[MotionClip]:
        tl = text.lower()
        sample = None
        for act, samples in self._index.items():
            if act in tl:
                sample = random.choice(samples)
                break
        if sample is None and self._samples:
            sample = random.choice(self._samples)
        if sample is None:
            return None
        raw = self._load_raw_joints(sample.sample_id, max_frames)
        return MotionClip(action=text, frames=sample.motion[:max_frames],
                          source="retrieved", raw_joints=raw)

    def _load_raw_joints(self, sample_id: str, max_frames: int) -> "np.ndarray | None":
        """Load (T, 21, 3) raw joint positions from new_joints/ if available."""
        import pathlib
        candidates = [
            pathlib.Path(self._data_dir) / "new_joints" / f"{sample_id}.npy",
        ]
        for p in candidates:
            if p.exists():
                arr = np.load(p)
                if arr.ndim == 3 and arr.shape[1] == 21:
                    return arr[:max_frames]
        return None


class SSMMotionModel:
    """Trained SSM model backend."""

    def __init__(self, checkpoint_path: str = DEFAULT_SSM_CHECKPOINT,
                 data_dir: str = DEFAULT_DATA_DIR):
        self.model  = None
        self._vocab: dict = {}
        self._device     = None
        self._data_dir   = data_dir
        self._load(checkpoint_path)

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            import sys
            import torch
            from src.modules.m4_motion_generator.train import TextToMotionSSM, TrainingConfig
            # Legacy compat: checkpoint was pickled under old module path
            if "src.motion_generator" not in sys.modules:
                import src.modules.m4_motion_generator as _m4
                sys.modules["src.motion_generator"] = _m4
                sys.modules["src.motion_generator.train"] = __import__(
                    "src.modules.m4_motion_generator.train", fromlist=["train"],
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
            self.model  = TextToMotionSSM(cfg).to(self._device)
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
                std  = np.load(str(Path(self._data_dir) / "Std.npy"))
                motion = motion * (std + 1e-8) + mean
            except OSError:
                pass
        return MotionClip(action=text, frames=motion, source="generated")


class MotionGenerator:
    """Unified backend: semantic/keyword retrieval > SSM > placeholder."""

    def __init__(self, use_retrieval: bool = True, use_ssm: bool = True,
                 use_semantic: bool = True,
                 data_dir: str = DEFAULT_DATA_DIR,
                 checkpoint_path: str = DEFAULT_SSM_CHECKPOINT):
        self._retriever: Optional[MotionRetriever] = None
        self._ssm: Optional[SSMMotionModel] = SSMMotionModel(checkpoint_path, data_dir) if use_ssm else None

        if use_retrieval:
            self._retriever = self._build_retriever(data_dir, use_semantic)

    # -- public properties (for benchmarks & introspection) --

    @property
    def retriever(self) -> Optional[MotionRetriever]:
        return self._retriever

    @property
    def ssm_model(self) -> Optional[SSMMotionModel]:
        return self._ssm

    # -- private --

    @staticmethod
    def _build_retriever(data_dir: str, use_semantic: bool) -> MotionRetriever:
        """Create the best available retriever: SBERT > keyword."""
        if use_semantic:
            try:
                from .semantic_retriever import SemanticRetriever
                return SemanticRetriever(data_dir)
            except Exception as exc:
                log.warning("SemanticRetriever unavailable (%s) — using keyword retriever", exc)
        return MotionRetriever(data_dir)

    @profile_memory
    def generate(self, text: str, num_frames: int = 100, prefer: str = "retrieval") -> MotionClip:
        if prefer == "retrieval" and self._retriever:
            if clip := self._retriever.retrieve(text, num_frames):
                return clip
        if prefer in ("ssm", "retrieval") and self._ssm and self._ssm.model:
            if clip := self._ssm.generate(text, num_frames):
                return clip
        return self._placeholder(text, num_frames)

    @staticmethod
    def _placeholder(text: str, num_frames: int) -> MotionClip:
        t      = np.linspace(0, 4 * np.pi, num_frames)
        motion = np.zeros((num_frames, MOTION_DIM))
        motion[:, 0] = 0.01 * np.sin(t)
        motion[:, 2] = 0.1 + 0.05 * np.sin(t * 0.5)
        return MotionClip(action=text, frames=motion, source="placeholder")


def create_motion_generator(**kwargs) -> MotionGenerator:
    return MotionGenerator(**kwargs)

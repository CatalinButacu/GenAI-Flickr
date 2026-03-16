from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.shared.mem_profile import profile_memory
from .constants import DEFAULT_DATA_DIR, DEFAULT_SSM_CHECKPOINT, MOTION_DIM
from .models import MotionClip

log = logging.getLogger(__name__)


def _trace_io(module_name: str, inputs: dict, outputs: dict) -> None:
    """Log compact IO trace for a motion generation backend."""
    def _s(v):
        if isinstance(v, np.ndarray):
            return f"shape={list(v.shape)}"
        return v
    parts = ", ".join(f"{k}={_s(v)}" for k, v in inputs.items())
    msg = f"[{module_name}] IN: {parts}"
    if outputs:
        parts_out = ", ".join(f"{k}={_s(v)}" for k, v in outputs.items())
        msg += f" | OUT: {parts_out}"
    log.debug(msg)


class MotionGenerator:
    """Unified backend: AMASS sample retrieval > SSM > placeholder."""

    def __init__(self, use_retrieval: bool = True, use_ssm: bool = True,
                 use_semantic: bool = True,
                 data_dir: str = DEFAULT_DATA_DIR,
                 checkpoint_path: str = DEFAULT_SSM_CHECKPOINT):
        self.retriever = None
        self.ssm_model = None

        if use_retrieval:
            self.retriever = build_retriever(data_dir, use_semantic)

        if use_ssm:
            from .ssm_model import SSMMotionModel
            self.ssm_model = SSMMotionModel(checkpoint_path, data_dir)

    @profile_memory
    def generate(self, text: str, num_frames: int = 100,
                 prefer: str = "retrieval") -> MotionClip:
        log.info("[MotionGen] generate(%r, n=%d, prefer=%s)", text, num_frames, prefer)
        _trace_io("MotionGen.generate INPUT", {"text": text, "num_frames": num_frames, "prefer": prefer}, {})
        if prefer == "retrieval" and self.retriever:
            if clip := self.retriever.retrieve(text, num_frames):
                log.info("[MotionGen] retrieval hit: %d frames, source=%s",
                         clip.num_frames, clip.source)
                _trace_io("MotionGen.retrieve", {"text": text, "max_frames": num_frames}, {
                    "frames_shape": clip.frames.shape if clip.frames is not None else None,
                    "raw_joints_shape": clip.raw_joints.shape if clip.raw_joints is not None else None,
                    "fps": clip.fps, "source": clip.source,
                })
                return clip
            log.info("[MotionGen] retrieval miss — trying SSM")
        if prefer in ("ssm", "retrieval") and self.ssm_model and self.ssm_model.model:
            if clip := self.ssm_model.generate(text, num_frames):
                log.info("[MotionGen] SSM generated: %d frames, source=%s",
                         clip.num_frames, clip.source)
                _trace_io("MotionGen.ssm", {"text": text, "num_frames": num_frames}, {
                    "frames_shape": clip.frames.shape if clip.frames is not None else None,
                    "fps": clip.fps, "source": clip.source,
                })
                return clip
            log.warning("[MotionGen] SSM generation failed — using placeholder")
        log.warning("[MotionGen] falling back to placeholder motion")
        clip = placeholder_motion(text, num_frames)
        _trace_io("MotionGen.placeholder", {"text": text, "num_frames": num_frames}, {
            "frames_shape": clip.frames.shape if clip.frames is not None else None,
            "fps": clip.fps, "source": clip.source,
        })
        return clip


def build_retriever(data_dir: str, use_semantic: bool):
    """Create an AMASS sample retriever, or None if data is missing."""
    if Path(data_dir).exists() and list(Path(data_dir).rglob("*.npz"))[:1]:
        from .amass_retriever import AMASSSampleRetriever
        log.info("[MotionGen] Using AMASSSampleRetriever from %s", data_dir)
        return AMASSSampleRetriever(amass_dir=data_dir, max_samples=200)
    log.warning("[MotionGen] AMASS data not found at %s", data_dir)
    return None


def placeholder_motion(text: str, num_frames: int) -> MotionClip:
    """Generate a simple sinusoidal placeholder motion."""
    t = np.linspace(0, 4 * np.pi, num_frames)
    motion = np.zeros((num_frames, MOTION_DIM))
    motion[:, 0] = 0.01 * np.sin(t)
    motion[:, 2] = 0.1 + 0.05 * np.sin(t * 0.5)
    return MotionClip(action=text, frames=motion, source="placeholder")


def create_motion_generator(**kwargs) -> MotionGenerator:
    return MotionGenerator(**kwargs)

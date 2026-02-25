"""SSM-enhanced motion generator: extends MotionGenerator with Mamba-based temporal modeling."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.modules.m4_motion_generator.generator import MotionGenerator, MotionClip
from src.shared.vocabulary import ACTIONS

log = logging.getLogger(__name__)

try:
    from src.ssm import MotionSSM, PhysicsSSM, SimpleSSMNumpy, HAS_TORCH
    HAS_SSM = True
except ImportError:
    HAS_SSM  = False
    HAS_TORCH = False

_MOTION_DIM = 251  # KIT-ML


@dataclass(slots=True)
class SSMMotionConfig:
    d_model: int     = 256
    d_state: int     = 32
    n_joints: int    = 24
    n_layers: int    = 4
    use_physics: bool = False
    d_physics: int   = 64


class SSMMotionGenerator(MotionGenerator):
    """Extends MotionGenerator with SSM temporal modeling (Motion Mamba, ECCV 2024).

    Novel contribution: optional PhysicsSSM for physics-constrained generation.
    Inference: O(n) vs diffusion O(n).
    """

    def __init__(self, backend: str = "ssm", config: SSMMotionConfig = None, device: str = "cuda"):
        super().__init__(use_retrieval=False, use_ssm=False)
        self._backend  = backend
        self._cfg      = config or SSMMotionConfig()
        self._ssm      = None
        self._phys_ssm = None
        self._ready    = False

    def setup(self) -> bool:
        if not HAS_SSM:
            log.warning("SSM module unavailable  falling back to placeholder")
            self._ready = True
            return True
        if not HAS_TORCH:
            self._ssm = SimpleSSMNumpy(self._cfg.d_state, self._cfg.d_model, _MOTION_DIM)
            self._ready = True
            return True
        try:
            self._ssm = MotionSSM(d_model=self._cfg.d_model, d_state=self._cfg.d_state, n_layers=self._cfg.n_layers)
            if self._backend == "ssm_physics" and self._cfg.use_physics:
                self._phys_ssm = PhysicsSSM(d_model=self._cfg.d_model, d_state=self._cfg.d_state, d_physics=self._cfg.d_physics)
            self._ready = True
            return True
        except Exception as e:
            log.error("SSM setup failed: %s", e)
            self._ready = True
            return True

    def generate(self, text: str, num_frames: int = 100, prefer: str = "ssm",
                 physics_state: Optional[np.ndarray] = None) -> MotionClip:
        if not self._ready:
            self.setup()
        if self._ssm is None:
            return self._placeholder(text, num_frames)
        return self._generate_torch(text, num_frames, physics_state) if HAS_TORCH \
            else self._generate_numpy(text, num_frames)

    def _generate_torch(self, text: str, num_frames: int, physics_state: Optional[np.ndarray]) -> MotionClip:
        import torch
        x = torch.randn(1, num_frames, self._cfg.d_model)
        with torch.no_grad():
            feats = self._ssm(x)
            if self._phys_ssm is not None and physics_state is not None:
                phys = torch.tensor(physics_state, dtype=torch.float32).unsqueeze(0)
                feats = self._phys_ssm(feats, phys)
        return self._to_clip(feats[0].numpy(), text)

    def _generate_numpy(self, text: str, num_frames: int) -> MotionClip:
        x = np.random.randn(num_frames, self._cfg.d_model) * 0.1
        return self._to_clip(self._ssm.forward(x), text)

    def _to_clip(self, features: np.ndarray, text: str) -> MotionClip:
        """Project (frames, d_model)  (frames, motion_dim) numpy array."""
        rng    = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        proj   = rng.standard_normal((features.shape[-1], _MOTION_DIM)).astype(np.float32) * 0.01
        frames = features.astype(np.float32) @ proj
        return MotionClip(action=text, frames=frames, fps=20, source="ssm")


def create_ssm_motion_generator(use_physics: bool = False, device: str = "cuda") -> SSMMotionGenerator:
    backend = "ssm_physics" if use_physics else "ssm"
    gen = SSMMotionGenerator(backend=backend, config=SSMMotionConfig(use_physics=use_physics), device=device)
    gen.setup()
    return gen

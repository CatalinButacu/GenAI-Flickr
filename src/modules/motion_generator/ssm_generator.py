"""
#WHERE
    Used by pipeline.py for SSM-enhanced motion generation with
    physics and temporal consistency layers.

#WHAT
    SSMMotionGenerator — extends MotionGenerator with Mamba-based temporal
    modeling, physics SSM for constraint enforcement, and temporal SSM for
    frame-to-frame consistency.

    Now loads trained PhysicsSSM checkpoint when available (checkpoints/physics_ssm/best_model.pt),
    falling back to random init only if no checkpoint exists.

#INPUT
    Text action description, optional SSM layer type.

#OUTPUT
    MotionClip with SSM-refined (T, 251) motion features.
"""

import logging
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.modules.motion_generator.generator import MotionGenerator
from src.modules.motion_generator.models import MotionClip
from src.modules.motion_generator.constants import MOTION_DIM
from src.shared.vocabulary import ACTIONS

log = logging.getLogger(__name__)

from src.modules.motion_generator.ssm import MotionSSM, PhysicsSSM, SimpleSSMNumpy, HAS_TORCH

from src.shared.constants import DEFAULT_PHYSICS_SSM_CHECKPOINT


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

    Loads trained weights from checkpoint when available.
    """

    def __init__(self, backend: str = "ssm", config: SSMMotionConfig = None,
                 device: str = "cuda",
                 checkpoint_path: str = DEFAULT_PHYSICS_SSM_CHECKPOINT):
        super().__init__(use_retrieval=False, use_ssm=False)
        self._backend  = backend
        self._cfg      = config or SSMMotionConfig()
        self._ssm      = None
        self._phys_ssm = None
        self._projector = None
        self._device_str = device
        self._checkpoint_path = checkpoint_path
        self._ready    = False

    def setup(self) -> bool:
        if not HAS_TORCH:
            self._ssm = SimpleSSMNumpy(self._cfg.d_state, self._cfg.d_model, MOTION_DIM)
            self._ready = True
            return True
        try:
            import torch
            self._ssm = MotionSSM(
                d_model=self._cfg.d_model, d_state=self._cfg.d_state,
                n_layers=self._cfg.n_layers,
            )
            if self._backend == "ssm_physics" and self._cfg.use_physics:
                self._phys_ssm = PhysicsSSM(
                    d_model=self._cfg.d_model, d_state=self._cfg.d_state,
                    d_physics=self._cfg.d_physics,
                )

            # Try loading trained checkpoint
            if os.path.exists(self._checkpoint_path):
                self._load_checkpoint(self._checkpoint_path)
            else:
                log.warning(
                    "PhysicsSSM checkpoint not found at %s — using random init",
                    self._checkpoint_path,
                )

            self._ready = True
            return True
        except Exception as e:
            log.error("SSM setup failed: %s", e)
            self._ready = False
            return False

    def _load_checkpoint(self, path: str) -> None:
        """Load trained MotionSSM + PhysicsSSM + MotionProjector weights."""
        import torch
        from src.modules.motion_generator.physics_trainer import MotionProjector

        device = torch.device(self._device_str if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)

        # Restore MotionSSM weights
        if "motion_ssm_state_dict" in ck and self._ssm is not None:
            self._ssm.load_state_dict(ck["motion_ssm_state_dict"], strict=False)
            self._ssm.eval()
            log.info("MotionSSM weights loaded from checkpoint")

        # Restore PhysicsSSM weights
        if "physics_ssm_state_dict" in ck and self._phys_ssm is not None:
            self._phys_ssm.load_state_dict(ck["physics_ssm_state_dict"], strict=False)
            self._phys_ssm.eval()
            log.info("PhysicsSSM weights loaded from checkpoint")

        # Restore learned MotionProjector (replaces random projection)
        if "projector_state_dict" in ck:
            self._projector = MotionProjector(
                d_model=self._cfg.d_model, motion_dim=MOTION_DIM,
            ).to(device)
            self._projector.load_state_dict(ck["projector_state_dict"], strict=False)
            self._projector.eval()
            log.info("MotionProjector loaded — using learned projection")

        log.info(
            "PhysicsSSM checkpoint loaded: epoch=%s val_loss=%s",
            ck.get("epoch", "?"), ck.get("val_loss", "?"),
        )

    def generate(self, text: str, num_frames: int = 100, prefer: str = "ssm",
                 physics_state: Optional[np.ndarray] = None) -> MotionClip:
        if not self._ready:
            self.setup()
        if self._ssm is None:
            return self._placeholder(text, num_frames)
        return self._generate_torch(text, num_frames, physics_state) if HAS_TORCH \
            else self._generate_numpy(text, num_frames)

    def _generate_torch(self, text: str, num_frames: int,
                        physics_state: Optional[np.ndarray]) -> MotionClip:
        import torch
        device = next(self._ssm.parameters()).device

        # If we have a learned projector, encode seed motion; otherwise random
        if self._projector is not None:
            # Use zero-init motion as seed (the model has learned a prior)
            seed = torch.zeros(1, num_frames, MOTION_DIM, device=device)
            x = self._projector.encode(seed)
        else:
            x = torch.randn(1, num_frames, self._cfg.d_model, device=device)

        with torch.no_grad():
            feats = self._ssm(x)
            if self._phys_ssm is not None and physics_state is not None:
                phys = torch.tensor(
                    physics_state, dtype=torch.float32, device=device,
                ).unsqueeze(0)
                feats = self._phys_ssm(feats, phys)

        # Decode to motion space
        if self._projector is not None:
            with torch.no_grad():
                motion = self._projector.decode(feats)
            return MotionClip(
                action=text, frames=motion[0].cpu().numpy(),
                fps=20, source="physics_ssm",
            )
        return self._to_clip(feats[0].cpu().numpy(), text)

    def _generate_numpy(self, text: str, num_frames: int) -> MotionClip:
        rng = np.random.default_rng(seed=abs(hash(text)) % (2 ** 31))
        x = rng.standard_normal((num_frames, self._cfg.d_model)) * 0.1
        return self._to_clip(self._ssm.forward(x), text)

    def _to_clip(self, features: np.ndarray, text: str) -> MotionClip:
        """Project (frames, d_model) → (frames, motion_dim) — fallback random."""
        rng    = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        proj   = rng.standard_normal((features.shape[-1], MOTION_DIM)).astype(np.float32) * 0.01
        frames = features.astype(np.float32) @ proj
        return MotionClip(action=text, frames=frames, fps=20, source="ssm")

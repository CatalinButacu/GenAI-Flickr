from __future__ import annotations

import logging
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass

from src.modules.motion.generator import MotionGenerator, placeholder_motion
from src.modules.motion.models import MotionClip
from src.modules.motion.constants import MOTION_DIM, MOTION_FPS
from src.modules.motion.nn_models import MotionProjector
from src.shared.vocabulary import ACTIONS

log = logging.getLogger(__name__)

from src.modules.motion.ssm import MotionSSM, PhysicsSSM

from src.shared.constants import (
    DEFAULT_PHYSICS_SSM_CHECKPOINT, SSM_D_MODEL, SSM_D_STATE,
    SSM_N_LAYERS, N_JOINTS, D_PHYSICS,
)


@dataclass(slots=True)
class SSMMotionConfig:
    d_model: int     = SSM_D_MODEL
    d_state: int     = SSM_D_STATE
    n_joints: int    = N_JOINTS
    n_layers: int    = SSM_N_LAYERS
    use_physics: bool = False
    d_physics: int   = D_PHYSICS


class SSMMotionGenerator(MotionGenerator):
    """Extends MotionGenerator with SSM temporal modeling (Motion Mamba, ECCV 2024).

    Novel contribution: optional PhysicsSSM for physics-constrained generation.
    Inference: O(n) vs diffusion O(n).

    Loads trained weights from checkpoint when available.
    """

    def __init__(self, backend: str = "ssm", config: SSMMotionConfig | None = None,
                 device: str = "cuda",
                 checkpoint_path: str = DEFAULT_PHYSICS_SSM_CHECKPOINT):
        super().__init__(use_retrieval=False, use_ssm=False)
        self.backend  = backend
        self.cfg      = config or SSMMotionConfig()
        self.ssm      = None
        self.phys_ssm = None
        self.projector = None
        self.device_str = device
        self.checkpoint_path = checkpoint_path
        self.ready    = False

    def setup(self) -> bool:
        try:
            self._create_ssm_models()
            if os.path.exists(self.checkpoint_path):
                self.load_checkpoint(self.checkpoint_path)
            else:
                log.warning("[SSMGen] checkpoint not found at %s — using random init",
                            self.checkpoint_path)
            self.ready = True
            log.info("[SSMGen] setup complete: ssm=%s, phys_ssm=%s, projector=%s",
                     self.ssm is not None, self.phys_ssm is not None,
                     self.projector is not None)
            return True
        except (OSError, RuntimeError) as e:
            log.error("[SSMGen] setup failed: %s", e)
            self.ready = False
            return False

    def _create_ssm_models(self) -> None:
        """Create MotionSSM and optionally PhysicsSSM."""
        self.ssm = MotionSSM(
            d_model=self.cfg.d_model, d_state=self.cfg.d_state,
            n_layers=self.cfg.n_layers,
        )
        log.info("[SSMGen] MotionSSM created: d_model=%d, d_state=%d, n_layers=%d",
                 self.cfg.d_model, self.cfg.d_state, self.cfg.n_layers)
        if self.backend == "ssm_physics" and self.cfg.use_physics:
            self.phys_ssm = PhysicsSSM(
                d_model=self.cfg.d_model, d_state=self.cfg.d_state,
                d_physics=self.cfg.d_physics,
            )
            log.info("[SSMGen] PhysicsSSM created: d_physics=%d, sigmoid gate active",
                     self.cfg.d_physics)
        else:
            log.info("[SSMGen] PhysicsSSM NOT created (backend=%r, use_physics=%s)",
                     self.backend, self.cfg.use_physics)

    def load_checkpoint(self, path: str) -> None:
        """Load trained MotionSSM + PhysicsSSM + MotionProjector weights."""
        device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device, weights_only=False)

        # Restore MotionSSM weights and move to device
        if "motion_ssm_state_dict" in ck and self.ssm is not None:
            self.ssm.load_state_dict(ck["motion_ssm_state_dict"], strict=False)
            self.ssm = self.ssm.to(device)
            self.ssm.eval()
            log.info("MotionSSM weights loaded from checkpoint → %s", device)

        # Restore PhysicsSSM weights and move to device
        if "physics_ssm_state_dict" in ck and self.phys_ssm is not None:
            self.phys_ssm.load_state_dict(ck["physics_ssm_state_dict"], strict=False)
            self.phys_ssm = self.phys_ssm.to(device)
            self.phys_ssm.eval()
            log.info("PhysicsSSM weights loaded from checkpoint → %s", device)

        # Restore learned MotionProjector (replaces random projection)
        if "projector_state_dict" in ck:
            self.projector = MotionProjector(
                d_model=self.cfg.d_model, motion_dim=MOTION_DIM,
            ).to(device)
            self.projector.load_state_dict(ck["projector_state_dict"], strict=False)
            self.projector.eval()
            log.info("MotionProjector loaded — using learned projection → %s", device)

        log.info(
            "PhysicsSSM checkpoint loaded: epoch=%s val_loss=%s",
            ck.get("epoch", "?"), ck.get("val_loss", "?"),
        )

    def generate(self, text: str, num_frames: int = 100, prefer: str = "ssm",
                 physics_state: np.ndarray | None = None) -> MotionClip:
        log.info("[SSMGen] generate(%r, n=%d, physics=%s)",
                 text[:60], num_frames,
                 physics_state.shape if physics_state is not None else "None")
        if not self.ready:
            self.setup()
        if self.ssm is None:
            log.warning("[SSMGen] SSM not available — returning placeholder")
            return placeholder_motion(text, num_frames)
        return self.generate_torch(text, num_frames, physics_state)

    def generate_torch(self, text: str, num_frames: int,
                        physics_state: np.ndarray | None) -> MotionClip:
        assert self.ssm is not None  # guaranteed by caller
        device = next(self.ssm.parameters()).device
        x = self._prepare_seed(num_frames, device)

        with torch.no_grad():
            feats = self.ssm(x)
            log.info("[SSMGen] MotionSSM output: %s", list(feats.shape))
            feats = self._apply_physics_gate(feats, physics_state, device)

        return self._decode_to_clip(feats, text)

    def _prepare_seed(self, num_frames, device):
        """Prepare initial seed tensor for SSM generation."""
        if self.projector is not None:
            seed = torch.zeros(1, num_frames, MOTION_DIM, device=device)
            x = self.projector.encode(seed)
            log.info("[SSMGen] using learned projector — seed encoded to %s", list(x.shape))
            return x
        x = torch.randn(1, num_frames, self.cfg.d_model, device=device)
        log.info("[SSMGen] no projector — using random seed %s", list(x.shape))
        return x

    def _apply_physics_gate(self, feats, physics_state, device):
        """Apply PhysicsSSM gate if both model and state are available."""
        if self.phys_ssm is not None and physics_state is not None:
            phys = torch.tensor(physics_state, dtype=torch.float32, device=device).unsqueeze(0)
            feats = self.phys_ssm(feats, phys)
            log.info("[SSMGen] PhysicsSSM gate applied — physics_state: %s", list(phys.shape))
        elif self.phys_ssm is None and physics_state is not None:
            log.warning("[SSMGen] physics_state provided but PhysicsSSM is None — gate NOT applied!")
        elif self.phys_ssm is not None and physics_state is None:
            log.warning("[SSMGen] PhysicsSSM loaded but no physics_state — running without gate")
        return feats

    def _decode_to_clip(self, feats, text) -> MotionClip:
        """Decode SSM features to a MotionClip."""
        if self.projector is not None:
            with torch.no_grad():
                motion = self.projector.decode(feats)
            log.info("[SSMGen] decoded → motion shape=%s, range=[%.4f, %.4f]",
                     list(motion.shape), float(motion.min()), float(motion.max()))
            return MotionClip(action=text, frames=motion[0].cpu().numpy(),
                              fps=MOTION_FPS, source="physics_ssm")
        log.info("[SSMGen] no projector — using random fallback projection")
        return self.to_clip(feats[0].cpu().numpy(), text)

    def to_clip(self, features: np.ndarray, text: str) -> MotionClip:
        """Project (frames, d_model) → (frames, motion_dim) — fallback random."""
        rng    = np.random.default_rng(seed=abs(hash(text)) % (2**31))
        proj   = rng.standard_normal((features.shape[-1], MOTION_DIM)).astype(np.float32) * 0.01
        frames = features.astype(np.float32) @ proj
        return MotionClip(action=text, frames=frames, fps=MOTION_FPS, source="ssm")

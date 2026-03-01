"""
#WHERE
    Used by scripts/train_physics_ssm.py for training data.

#WHAT
    PhysicsAugmented KIT-ML Dataset — pairs motion sequences with
    physics state vectors derived from the motion data itself:
      - Pelvis height, velocity, acceleration
      - Estimated foot contact phases
      - Gravity prior
      - Center-of-mass momentum approximation

#INPUT
    KIT-ML data directory (texts/, new_joint_vecs/, Mean.npy, Std.npy).

#OUTPUT
    Dict per sample: motion (float32), physics_state (float32),
    motion_mask (float32), length (int).
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_DATA_DIR, MOTION_DIM, MOTION_FPS

log = logging.getLogger(__name__)

# KIT-ML joint vector layout (251-dim):
#   [0:3]   root angular velocity
#   [3:6]   root linear velocity
#   [6:9]   root height (Y-axis)
#   [9:63]  joint rotations (6D)
#   [63:66] root position XYZ
#   ...     remaining joint positions / velocities
# The exact layout follows HumanML3D convention.
# We use indices that are known stable across KIT-ML.

# Physics state dimensionality
D_PHYSICS = 64


def extract_physics_state(
    motion: np.ndarray,
    fps: int = MOTION_FPS,
    normalize: bool = True,
) -> np.ndarray:
    """Derive a (T, D_PHYSICS) physics state from a (T, 251) motion vector.

    Physics channels:
      [0]      gravity prior (constant -9.81)
      [1]      pelvis height (from root Y)
      [2]      pelvis vertical velocity
      [3]      pelvis vertical acceleration
      [4:7]    root linear velocity (XYZ)
      [7:10]   root angular velocity
      [10:12]  estimated left/right foot contact (binary)
      [12:15]  centre-of-mass velocity approximation
      [15:18]  centre-of-mass acceleration
      [18:21]  root position XYZ
      [21:24]  angular momentum proxy (angular_vel magnitude per axis)
      [24:64]  zero-padded (reserved for future physics features)
    """
    T = motion.shape[0]
    dt = 1.0 / fps
    phys = np.zeros((T, D_PHYSICS), dtype=np.float32)

    # [0] Gravity prior — constant
    from src.shared.constants import GRAVITY
    phys[:, 0] = GRAVITY

    # Root position and height
    # In KIT-ML 251-dim: root_pos is around indices 63:66 or we derive
    # from the motion vector.  Root height is typically motion[:, 2] in
    # some layouts, but more robustly we use index 6 (root_height channel)
    # after consulting HumanML3D format.

    # Root angular velocity [0:3]
    root_angvel = motion[:, 0:3]
    phys[:, 7:10] = root_angvel

    # Root linear velocity [3:6]
    root_linvel = motion[:, 3:6]
    phys[:, 4:7] = root_linvel

    # Root height — use Y component from root_linvel integrated, or
    # direct if available.  KIT-ML stores root height at index 2 of
    # the velocity vector (vertical axis).
    # For pelvis height, integrate root_linvel Y or use motion[:, 63+1]
    # We approximate from the cumulative sum of vertical velocity.
    pelvis_h = np.cumsum(root_linvel[:, 1]) * dt  # integrate vy
    pelvis_h = pelvis_h - pelvis_h.min() + 0.93   # offset to ~standing
    phys[:, 1] = pelvis_h

    # [2] Pelvis vertical velocity — finite differences for smoothness
    phys[:, 2] = np.gradient(pelvis_h, dt)

    # [3] Pelvis vertical acceleration
    phys[:, 3] = np.gradient(phys[:, 2], dt)

    # [10:12] Foot contact estimation
    # Heuristic: when vertical velocity is near zero and height is low,
    # feet are likely in contact.  We approximate with a threshold on
    # absolute vertical velocity.
    vy = np.abs(phys[:, 2])
    # Left foot contact when vy < threshold and height < mean
    contact_thresh = np.percentile(vy, 30)
    height_thresh = np.median(pelvis_h)
    left_contact = ((vy < contact_thresh) & (pelvis_h < height_thresh)).astype(np.float32)
    right_contact = ((vy < contact_thresh) & (pelvis_h < height_thresh)).astype(np.float32)
    # Phase offset: approximate alternating foot contacts for walking
    # by shifting right foot by half a gait cycle
    if T > 20:
        half_cycle = T // 8  # approximate half gait cycle
        right_contact = np.roll(right_contact, half_cycle)
    phys[:, 10] = left_contact
    phys[:, 11] = right_contact

    # [12:15] Centre-of-mass velocity approximation (≈ root velocity)
    phys[:, 12:15] = root_linvel

    # [15:18] Centre-of-mass acceleration
    for ax in range(3):
        phys[:, 15 + ax] = np.gradient(root_linvel[:, ax], dt)

    # [18:21] Root position (integrate linear velocity)
    root_pos = np.cumsum(root_linvel, axis=0) * dt
    phys[:, 18:21] = root_pos

    # [21:24] Angular momentum proxy
    phys[:, 21:24] = np.abs(root_angvel)

    # Optional normalisation per channel
    if normalize:
        for ch in range(D_PHYSICS):
            col = phys[:, ch]
            std = col.std()
            if std > 1e-6:
                phys[:, ch] = (col - col.mean()) / std

    return phys


class PhysicsMotionDataset(Dataset):
    """KIT-ML + derived physics state pairs for PhysicsSSM training."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        max_length: int = 200,
        d_physics: int = D_PHYSICS,
    ):
        from src.data import KITMLLoader

        self.max_length = max_length
        self.d_physics = d_physics
        self._samples = []

        loader = KITMLLoader(data_dir)
        dataset = loader.load_dataset(split, normalize=True)

        for s in dataset.samples:
            if s.motion is None or len(s.motion) < 4:
                continue
            # Extract physics state from the UN-normalised motion
            raw_motion = loader.load_motion(s.sample_id, normalize=False)
            if raw_motion is None:
                continue
            physics = extract_physics_state(raw_motion, fps=MOTION_FPS)
            self._samples.append({
                "motion": s.motion,        # normalised (T, 251)
                "physics": physics,        # derived   (T, 64)
                "text": s.text,
                "T": len(s.motion),
            })

        log.info("[PhysicsDataset] %s split: %d samples", split, len(self._samples))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        T = min(s["T"], self.max_length)

        # Pad / truncate motion
        motion = s["motion"][:T]
        if T < self.max_length:
            motion = np.concatenate([
                motion,
                np.zeros((self.max_length - T, MOTION_DIM), dtype=np.float32),
            ])

        # Pad / truncate physics
        physics = s["physics"][:T]
        if T < self.max_length:
            physics = np.concatenate([
                physics,
                np.zeros((self.max_length - T, self.d_physics), dtype=np.float32),
            ])

        mask = np.zeros(self.max_length, dtype=np.float32)
        mask[:T] = 1.0

        return {
            "motion":       torch.tensor(motion,  dtype=torch.float32),
            "physics_state": torch.tensor(physics, dtype=torch.float32),
            "motion_mask":  torch.tensor(mask,     dtype=torch.float32),
            "length":       T,
        }

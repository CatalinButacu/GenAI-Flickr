
from dataclasses import dataclass

import numpy as np

from src.shared.constants import N_BODY_JOINTS

from .constants import MOTION_FPS


@dataclass(slots=True)
class MotionClip:
    action: str
    frames: np.ndarray                      # (T, MOTION_DIM=168) SMPL-X pose params
    fps: int    = MOTION_FPS               # canonical: 30 fps (AMASS/InterX/PAHOI)
    source: str = "generated"
    # Body-only joint positions: (T, N_BODY_JOINTS=22, 3) in mm, Y-up.
    # N_BODY_JOINTS = pelvis + 21 body joints (excludes hands/jaw/eyes).
    raw_joints: "np.ndarray | None" = None
    betas: "np.ndarray | None" = None       # (16,) SMPL-X shape parameters

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    @property
    def num_frames(self) -> int:
        return len(self.frames)

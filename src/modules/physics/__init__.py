"""Physics module — simulation, body models, skeleton utilities."""

# Core
from .scene import Scene
from .camera import CameraConfig, CinematicCamera, FrameData
from .simulator import Simulator
from .humanoid import HumanoidBody, HumanoidConfig, load_humanoid

# Motion retargeting
from .motion_retarget import retarget_frame, retarget_sequence, pelvis_transform

# Skeleton utilities
from .physics_renderer import (
    PhysicsSkeletonRenderer, physics_links_to_skeleton, BONES,
    auto_orient_skeleton,
)

# Body models
from .body_model import BodyParams, BodyType, body_params_from_prompt
from .silhouette_renderer import SilhouetteSkeletonRenderer, SilhouetteProjector
from .smpl_body import (
    HAS_SMPL, SMPLBody, is_smpl_available,
    smpl_betas_from_body_params, render_smpl_silhouette,
    KIT_TO_SMPL, SMPL_PARENTS,
)

__all__ = [
    # Core
    'Scene', 'Simulator', 'CameraConfig', 'CinematicCamera', 'FrameData',
    'HumanoidBody', 'HumanoidConfig', 'load_humanoid',
    # Retargeting
    'retarget_frame', 'retarget_sequence', 'pelvis_transform',
    # Skeleton
    'PhysicsSkeletonRenderer', 'physics_links_to_skeleton', 'BONES',
    'auto_orient_skeleton',
    # Body models
    'BodyParams', 'BodyType', 'body_params_from_prompt',
    'SilhouetteSkeletonRenderer', 'SilhouetteProjector',
    'HAS_SMPL', 'SMPLBody', 'is_smpl_available',
    'smpl_betas_from_body_params', 'render_smpl_silhouette',
    'KIT_TO_SMPL', 'SMPL_PARENTS',
]

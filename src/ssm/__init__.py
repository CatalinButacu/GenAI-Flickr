"""
State Space Models (SSM) Module
===============================
Provides SSM layers for motion and video generation.

This module implements:
- S4 (Structured State Spaces) - Gu et al., ICLR 2022
- Mamba (Selective SSMs) - Gu & Dao, 2024
- MotionSSM - Inspired by Motion Mamba, ECCV 2024
- PhysicsSSM - Novel physics-constrained SSM (our contribution)
- TemporalConsistencySSM - For video frame consistency

Usage:
    from src.ssm import MambaLayer, MotionSSM, PhysicsSSM
    
    # Create a Mamba layer
    layer = MambaLayer(SSMConfig(d_model=256))
    output = layer(input_sequence)
    
    # Create physics-aware SSM
    physics_ssm = PhysicsSSM(d_model=256, d_physics=64)
    output = physics_ssm(motion, physics_state)

References:
    [1] S4: https://arxiv.org/abs/2111.00396
    [2] Mamba: https://arxiv.org/abs/2312.00752
    [3] Motion Mamba: https://arxiv.org/abs/2403.07487
"""

import torch

HAS_TORCH = True

from .core import (
    # Configuration
    SSMConfig,
    
    # Layers (PyTorch-based)
    S4Layer,
    MambaLayer,
    TemporalConsistencySSM,
    MotionSSM,
    PhysicsSSM,
    
    # NumPy fallback
    SimpleSSMNumpy,
    
    # Utilities
    create_ssm_layer,
    get_ssm_info,
)

__all__ = [
    # Config
    "SSMConfig",
    
    # Layers
    "S4Layer",
    "MambaLayer", 
    "TemporalConsistencySSM",
    "MotionSSM",
    "PhysicsSSM",
    
    # Fallback
    "SimpleSSMNumpy",
    
    # Utils
    "create_ssm_layer",
    "get_ssm_info",
    "HAS_TORCH",
]

# 3D Model Generator Module
"""
This module generates 3D models from text prompts or images.
Supports two backends:
- Shap-E: Text → 3D (direct, medium quality)
- TripoSR: Image → 3D (high quality, needs image)
"""

from .generator import ModelGenerator

__all__ = ['ModelGenerator']

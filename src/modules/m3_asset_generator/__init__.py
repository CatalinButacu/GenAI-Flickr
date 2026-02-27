"""
#WHERE
    Imported by pipeline.py, benchmarks, test_model_generator.py.

#WHAT
    3D Model Generator Module (Module 3) — generates 3D meshes from text
    prompts (Shap-E) or reference images (TripoSR) using Strategy pattern.

#INPUT
    Text prompt or image path, output directory, backend choice.

#OUTPUT
    GeneratedModel dataclass with .obj/.ply mesh file on disk.
"""

from .generator import ModelGenerator
from .models import GeneratedModel, GenerationBackend
from .backends import ShapEBackend, TripoSRBackend

__all__ = [
    "ModelGenerator", "GeneratedModel", "GenerationBackend",
    "ShapEBackend", "TripoSRBackend",
]

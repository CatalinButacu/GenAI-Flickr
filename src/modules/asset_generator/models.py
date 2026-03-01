"""
#WHERE
    Used by generator.py, pipeline.py, benchmarks, and test_model_generator.py.

#WHAT
    Data models for the 3D asset generator module: GeneratedModel dataclass
    and GenerationBackend protocol (Strategy pattern interface).

#INPUT
    Model metadata (name, paths, backend tag).

#OUTPUT
    GeneratedModel dataclass instance.
"""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(slots=True)
class GeneratedModel:
    name: str
    mesh_path: str
    preview_path: Optional[str] = None
    backend: str = "unknown"


class GenerationBackend(Protocol):
    """Strategy interface for 3D generation backends."""

    def setup(self) -> bool: ...
    def generate(self, prompt: str, output_dir: str, name: str) -> Optional[GeneratedModel]: ...

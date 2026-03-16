
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class GeneratedModel:
    name: str
    mesh_path: str
    preview_path: str | None = None
    backend: str = "unknown"


class GenerationBackend(Protocol):
    """Strategy interface for 3D generation backends."""

    def setup(self) -> bool: ...
    def generate(self, prompt: str, output_dir: str, name: str) -> GeneratedModel | None: ...

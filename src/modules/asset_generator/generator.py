"""
#WHERE
    Imported by pipeline.py, benchmarks, test_model_generator.py.
    Re-exports GeneratedModel, ShapEBackend, TripoSRBackend for backward compat.

#WHAT
    ModelGenerator orchestrator — selects a 3D generation backend (Shap-E or
    TripoSR) and delegates text-to-3D or image-to-3D generation.

#INPUT
    Backend name ("shap-e" | "triposr"), device, text prompt or image path.

#OUTPUT
    GeneratedModel with mesh file path on disk.
"""

import logging
from typing import Optional

from .models import GeneratedModel, GenerationBackend          # noqa: F401
from .backends import ShapEBackend, TripoSRBackend             # noqa: F401

log = logging.getLogger(__name__)


class ModelGenerator:
    BACKENDS = {
        "shap-e": ShapEBackend,
        "triposr": TripoSRBackend,
    }

    def __init__(self, backend: str = "shap-e", device: str = "cuda"):
        self.backend_name = backend
        self.device = device
        self._backend: Optional[GenerationBackend] = None
        self._is_setup = False

    def setup(self) -> bool:
        backend_class = self.BACKENDS.get(self.backend_name)
        if not backend_class:
            log.error("Unknown backend: %s", self.backend_name)
            return False
        self._backend = backend_class(device=self.device)
        self._is_setup = self._backend.setup()
        return self._is_setup

    def generate_from_text(self, prompt: str, output_dir: str, name: str = None,
                           guidance_scale: float = 15.0,
                           num_inference_steps: int = 64) -> Optional[GeneratedModel]:
        if self.backend_name != "shap-e":
            log.error("Text-to-3D requires Shap-E backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(prompt, output_dir, name,
                                      guidance_scale, num_inference_steps)

    def generate_from_image(self, image_path: str, output_dir: str,
                            name: str = None) -> Optional[GeneratedModel]:
        if self.backend_name != "triposr":
            log.error("Image-to-3D requires TripoSR backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(image_path, output_dir, name)


from __future__ import annotations

import logging

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
        self._backend: GenerationBackend | None = None
        self._is_setup = False

    def setup(self) -> bool:
        if not (backend_class := self.BACKENDS.get(self.backend_name)):
            log.error("Unknown backend: %s", self.backend_name)
            return False
        self._backend = backend_class(device=self.device)
        self._is_setup = self._backend.setup()
        return self._is_setup

    def generate_from_text(self, prompt: str, output_dir: str, name: str = None,
                           guidance_scale: float = 15.0,
                           num_inference_steps: int = 64) -> GeneratedModel | None:
        if self.backend_name != "shap-e":
            log.error("Text-to-3D requires Shap-E backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(prompt, output_dir, name,
                                      guidance_scale, num_inference_steps)

    def generate_from_image(self, image_path: str, output_dir: str,
                            name: str = None) -> GeneratedModel | None:
        if self.backend_name != "triposr":
            log.error("Image-to-3D requires TripoSR backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(image_path, output_dir, name)

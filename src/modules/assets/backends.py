
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

import torch
from diffusers import ShapEPipeline
from diffusers.utils import export_to_ply, export_to_obj
from PIL import Image

# TripoSR is an optional dependency — only available if installed
if importlib.util.find_spec("tsr") is not None:
    from tsr.system import TSR
else:
    TSR = None  # type: ignore[misc,assignment]

from .models import GeneratedModel

log = logging.getLogger(__name__)


class ShapEBackend:
    """Text-to-3D via OpenAI Shap-E diffusion pipeline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None
        self._is_ready = False

    def setup(self) -> bool:
        try:
            self._pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._is_ready = True
            log.info("Shap-E backend ready")
            return True
        except Exception as e:
            log.error("Shap-E setup failed: %s", e)
            return False

    def generate(self, prompt: str, output_dir: str, name: str = None,
                 guidance_scale: float = 15.0,
                 num_steps: int = 64) -> GeneratedModel | None:
        if not self._is_ready:
            log.error("Backend not ready")
            return None

        os.makedirs(output_dir, exist_ok=True)
        name = name or prompt.replace(" ", "_")[:30]

        with torch.no_grad():
            output = self._pipe(
                prompt, guidance_scale=guidance_scale,
                num_inference_steps=num_steps, output_type="mesh",
            )

        mesh = output.images[0]
        ply_path = os.path.join(output_dir, f"{name}.ply")
        export_to_ply(mesh, ply_path)

        obj_path = os.path.join(output_dir, f"{name}.obj")
        try:
            export_to_obj(mesh, obj_path)
            final_path = obj_path
        except Exception as exc:
            log.warning("OBJ export failed, falling back to PLY: %s", exc)
            final_path = ply_path

        log.info("Generated: %s", final_path)
        return GeneratedModel(name=name, mesh_path=final_path, backend="shap-e")


class TripoSRBackend:
    """Image-to-3D via Stability AI TripoSR."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._is_ready = False

    def setup(self) -> bool:
        try:
            self._model = TSR.from_pretrained(
                "stabilityai/TripoSR",
                config_name="config.yaml",
                weight_name="model.ckpt",
            )
            self._model.to(self.device)
            self._is_ready = True
            log.info("TripoSR backend ready")
            return True
        except ImportError:
            log.error("TripoSR not installed")
            return False
        except Exception as e:
            log.error("TripoSR setup failed: %s", e)
            return False

    def generate(self, image_path: str, output_dir: str,
                 name: str = None) -> GeneratedModel | None:
        if not self._is_ready:
            log.error("Backend not ready")
            return None
        if not os.path.exists(image_path):
            log.error("Image not found: %s", image_path)
            return None

        os.makedirs(output_dir, exist_ok=True)
        name = name or Path(image_path).stem

        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            scene_codes = self._model([image], device=self.device)

        meshes = self._model.extract_mesh(scene_codes)
        mesh = meshes[0]

        obj_path = os.path.join(output_dir, f"{name}.obj")
        mesh.export(obj_path)

        log.info("Generated: %s", obj_path)
        return GeneratedModel(name=name, mesh_path=obj_path, backend="triposr")

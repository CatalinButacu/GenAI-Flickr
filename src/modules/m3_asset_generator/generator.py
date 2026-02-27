"""3D Model Generator with Strategy pattern for multiple backends."""

import os
import logging
from pathlib import Path
from typing import Optional, Protocol
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(slots=True)
class GeneratedModel:
    name: str
    mesh_path: str
    preview_path: Optional[str] = None
    backend: str = "unknown"


class GenerationBackend(Protocol):
    def setup(self) -> bool: ...
    def generate(self, prompt: str, output_dir: str, name: str) -> Optional[GeneratedModel]: ...


class ShapEBackend:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._pipe = None
        self._is_ready = False
    
    def setup(self) -> bool:
        import torch
        from diffusers import ShapEPipeline

        try:
            self._pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            self._is_ready = True
            log.info("Shap-E backend ready")
            return True
        except Exception as e:
            log.error("Shap-E setup failed: %s", e)
            return False
    
    def generate(self, prompt: str, output_dir: str, name: str = None, 
                 guidance_scale: float = 15.0, num_steps: int = 64) -> Optional[GeneratedModel]:
        if not self._is_ready:
            log.error("Backend not ready")
            return None
        
        import torch
        from diffusers.utils import export_to_ply, export_to_obj
        
        os.makedirs(output_dir, exist_ok=True)
        name = name or prompt.replace(" ", "_")[:30]
        
        with torch.no_grad():
            output = self._pipe(prompt, guidance_scale=guidance_scale, 
                               num_inference_steps=num_steps, output_type="mesh")
        
        mesh = output.images[0]
        ply_path = os.path.join(output_dir, f"{name}.ply")
        export_to_ply(mesh, ply_path)
        
        obj_path = os.path.join(output_dir, f"{name}.obj")
        try:
            export_to_obj(mesh, obj_path)
            final_path = obj_path
        except Exception:
            final_path = ply_path
        
        log.info("Generated: %s", final_path)
        return GeneratedModel(name=name, mesh_path=final_path, backend="shap-e")


class TripoSRBackend:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None
        self._is_ready = False
    
    def setup(self) -> bool:
        try:
            from tsr.system import TSR
            
            self._model = TSR.from_pretrained(
                "stabilityai/TripoSR", 
                config_name="config.yaml", 
                weight_name="model.ckpt"
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
    
    def generate(self, image_path: str, output_dir: str, name: str = None) -> Optional[GeneratedModel]:
        if not self._is_ready:
            log.error("Backend not ready")
            return None
        
        if not os.path.exists(image_path):
            log.error("Image not found: %s", image_path)
            return None
        
        import torch
        from PIL import Image
        
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


class ModelGenerator:
    BACKENDS = {
        "shap-e": ShapEBackend,
        "triposr": TripoSRBackend
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
                          guidance_scale: float = 15.0, num_inference_steps: int = 64) -> Optional[GeneratedModel]:
        if self.backend_name != "shap-e":
            log.error("Text-to-3D requires Shap-E backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(prompt, output_dir, name, guidance_scale, num_inference_steps)
    
    def generate_from_image(self, image_path: str, output_dir: str, name: str = None) -> Optional[GeneratedModel]:
        if self.backend_name != "triposr":
            log.error("Image-to-3D requires TripoSR backend")
            return None
        if not self._is_setup:
            log.error("Call setup() first")
            return None
        return self._backend.generate(image_path, output_dir, name)

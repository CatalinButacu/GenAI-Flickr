"""
#WHERE
    Imported by m8_ai_enhancer/__init__.py, pipeline.py.

#WHAT
    VideoRenderer — legacy depth-conditioned ControlNet enhancement.
    Generates photorealistic frames from M5 depth maps.

#INPUT
    List[FrameData] with depth arrays, text prompt, device.

#OUTPUT
    List[EnhancedFrame] with original + enhanced RGB.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class EnhancedFrame:
    original_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    depth: np.ndarray
    timestamp: float


class VideoRenderer:
    """ControlNet-based per-frame renderer. Call setup() before use."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 controlnet_id: str = "lllyasviel/sd-controlnet-depth",
                 device: str = "cuda"):
        self.model_id      = model_id
        self.controlnet_id = controlnet_id
        self.device        = device
        self._pipe         = None
        self._ready        = False

    def setup(self) -> bool:
        try:
            import torch
            from diffusers import (StableDiffusionControlNetPipeline,
                                   ControlNetModel, UniPCMultistepScheduler)
            dtype      = torch.float16 if self.device == "cuda" else torch.float32
            controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=dtype)
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
            ).to(self.device)
            self._pipe.scheduler = UniPCMultistepScheduler.from_config(self._pipe.scheduler.config)
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
            self._ready = True
            log.info("VideoRenderer ready")
            return True
        except Exception as e:
            log.error("VideoRenderer setup failed: %s", e)
            return False

    def enhance_frame(self, depth_map: np.ndarray, prompt: str,
                      negative_prompt: str = "blurry, low quality",
                      num_steps: int = 20, guidance_scale: float = 7.5,
                      controlnet_scale: float = 1.0) -> np.ndarray:
        if not self._ready:
            return depth_map
        from PIL import Image
        import torch
        depth_rgb = np.stack([depth_map] * 3, axis=-1) if depth_map.ndim == 2 else depth_map
        with torch.no_grad():
            result = self._pipe(
                prompt=prompt, negative_prompt=negative_prompt,
                image=Image.fromarray(depth_rgb.astype(np.uint8)),
                num_inference_steps=num_steps, guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
            ).images[0]
        return np.array(result)

    def create_video(self, frames: List[EnhancedFrame], output_path: str,
                     fps: int = 24, comparison: bool = True) -> str:
        import imageio
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264")
        for frame in frames:
            img = np.concatenate([frame.original_rgb, frame.enhanced_rgb], axis=1) if comparison else frame.enhanced_rgb
            writer.append_data(img)
        writer.close()
        log.info("Video saved: %s", output_path)
        return output_path

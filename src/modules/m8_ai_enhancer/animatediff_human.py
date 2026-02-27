"""
AnimateDiff + ControlNet OpenPose  –  Temporally-consistent human video.

Architecture
~~~~~~~~~~~~
This builds on top of ``controlnet_human.py`` but replaces per-frame
inference with **batch 8-16 frame generation** using AnimateDiff's
temporal attention layers.  The key difference:

    Per-frame ControlNet (old):
        Frame 0 → diffusion → RGB₀
        Frame 1 → diffusion → RGB₁     ← no awareness of Frame 0!
        …flickering…

    AnimateDiff + ControlNet (this module):
        [Frame 0, Frame 1, …, Frame 7] → diffusion with cross-frame attn → [RGB₀…RGB₇]
        All frames share temporal attention → consistent identity/clothing/skin

Paper
~~~~~
Guo et al. 2023, "AnimateDiff: Animate Your Personalized Text-to-Image
Diffusion Models without Specific Tuning" (arXiv:2307.04725)

VRAM
~~~~
SD 1.5 + ControlNet + MotionAdapter in float16 with CPU offload ≈ 4.0-4.5 GB.
Borderline on RTX 3050 (4 GB).  Start with 8 frames at 512×512.

First run downloads the motion adapter (~1.8 GB) from HuggingFace:
    guoyww/animatediff-motion-adapter-v1-5-3
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)


class AnimateDiffHumanRenderer:
    """Generates temporally-consistent photorealistic human video clips.

    Uses ``AnimateDiffControlNetPipeline`` from ``diffusers ≥ 0.25``:
      - SD 1.5 backbone (already cached, ~4.3 GB)
      - ControlNet OpenPose conditioning (already cached, ~700 MB)
      - MotionAdapter temporal attention (downloads ~1.8 GB on first use)

    Call :meth:`setup` once, then :meth:`render_batch` for groups of 8 frames.
    """

    # Default model IDs  ────────────────────────────────────────────────
    SD_MODEL = "runwayml/stable-diffusion-v1-5"
    CN_MODEL = "lllyasviel/sd-controlnet-openpose"
    MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-3"

    def __init__(
        self,
        device: str = "cuda",
        prompt: str = (
            "a person walking naturally, full body visible, "
            "photorealistic, 8k uhd, cinematic lighting, "
            "professional photography, studio backdrop, "
            "detailed clothing, natural skin texture"
        ),
        negative_prompt: str = (
            "blurry, cartoon, sketch, anime, deformed, extra limbs, "
            "bad anatomy, low quality, watermark, text, logo, "
            "cropped, out of frame, nude, nsfw, disfigured"
        ),
        num_steps: int = 15,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 1.0,
        seed: int = 42,
        batch_size: int = 4,
        width: int = 384,
        height: int = 384,
    ) -> None:
        self.device = device
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale
        self.seed = seed
        self.batch_size = batch_size
        self.width = width
        self.height = height

        self._pipe = None
        self._ready = False

    # ─────────────────────────────────────────────────────────────────────
    def setup(self) -> bool:
        """Load SD 1.5 + ControlNet OpenPose + MotionAdapter.

        Downloads ~1.8 GB on first run (motion adapter).
        Returns True on success, False on failure.
        """
        try:
            import torch
            from diffusers.pipelines.animatediff.pipeline_animatediff_controlnet import (
                AnimateDiffControlNetPipeline,
            )
            from diffusers.models.unets.unet_motion_model import MotionAdapter
            from diffusers.models.controlnets.controlnet import ControlNetModel
            from diffusers.schedulers.scheduling_unipc_multistep import (
                UniPCMultistepScheduler,
            )

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            log.info("Loading MotionAdapter (~1.8 GB on first run)…")
            adapter = MotionAdapter.from_pretrained(
                self.MOTION_ADAPTER,
                torch_dtype=dtype,
            )

            log.info("Loading ControlNet OpenPose…")
            controlnet = ControlNetModel.from_pretrained(
                self.CN_MODEL,
                torch_dtype=dtype,
            )

            log.info("Loading SD 1.5 + assembling AnimateDiff pipeline…")
            pipe = AnimateDiffControlNetPipeline.from_pretrained(
                self.SD_MODEL,
                motion_adapter=adapter,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
            )

            # VRAM optimisation — critical for 4 GB
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
            else:
                pipe = pipe.to(self.device)

            pipe.enable_attention_slicing(1)

            try:
                pipe.enable_xformers_memory_efficient_attention()
                log.info("xformers memory-efficient attention enabled")
            except Exception:
                pass

            self._pipe = pipe
            self._ready = True
            log.info("AnimateDiffHumanRenderer ready on %s", self.device)
            return True

        except Exception as exc:
            log.error("AnimateDiffHumanRenderer setup failed: %s", exc)
            return False

    # ─────────────────────────────────────────────────────────────────────
    def render_batch(
        self,
        skeleton_images: List[np.ndarray],
        prompt_override: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Render a batch of skeleton images as temporally-consistent frames.

        Parameters
        ----------
        skeleton_images : list of ndarray (H, W, 3) uint8
            OpenPose-style skeleton images (from ``SkeletonProjector``).
            Length should be ≤ ``self.batch_size`` (default 8).
        prompt_override : str, optional
            Override the default prompt.

        Returns
        -------
        list of ndarray (H, W, 3) uint8 — photorealistic RGB frames.
        """
        if not self._ready or self._pipe is None:
            log.warning("AnimateDiff not loaded — returning skeleton images")
            return skeleton_images

        import torch
        from PIL import Image

        pipe = self._pipe
        prompt = prompt_override or self.prompt
        n_frames = len(skeleton_images)

        # Resize skeleton images to target resolution to save VRAM
        pil_images: list = [
            Image.fromarray(img).resize(
                (self.width, self.height), Image.Resampling.LANCZOS,
            )
            for img in skeleton_images
        ]

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        log.info("[M8] AnimateDiff inference: %d frames @ %dx%d, %d steps\u2026",
                 n_frames, self.width, self.height, self.num_steps)

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                conditioning_frames=pil_images,  # type: ignore[arg-type]
                num_frames=n_frames,
                width=self.width,
                height=self.height,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_scale,
                generator=generator,
            )

        # output.frames is a list of lists of PIL Images
        result_frames = output.frames[0]  # type: ignore[union-attr]

        # Upscale back to original skeleton resolution if needed
        orig_h, orig_w = skeleton_images[0].shape[:2]
        results = []
        for f in result_frames:
            arr = np.array(f)
            if arr.shape[0] != orig_h or arr.shape[1] != orig_w:
                arr = np.array(
                    Image.fromarray(arr).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
                )
            results.append(arr)
        return results

    # ─────────────────────────────────────────────────────────────────────
    def render_sequence(
        self,
        skeleton_images: List[np.ndarray],
        prompt_override: Optional[str] = None,
        progress_callback: Optional[object] = None,
    ) -> List[np.ndarray]:
        """Render a full sequence by splitting into batches.

        Handles sequences longer than ``batch_size`` by splitting into
        overlapping windows — the last frame of batch N is the same pose
        as the first frame of batch N+1, ensuring smooth transitions.

        Returns list of (H, W, 3) uint8 RGB frames.
        """
        if not self._ready:
            log.warning("AnimateDiff not loaded — returning skeleton images")
            return skeleton_images

        total = len(skeleton_images)
        bs = self.batch_size

        if total <= bs:
            return self.render_batch(skeleton_images, prompt_override)

        # Split into overlapping windows
        all_frames: List[np.ndarray] = []
        batch_idx = 0

        i = 0
        while i < total:
            end = min(i + bs, total)
            batch = skeleton_images[i:end]

            # Pad if too short for AnimateDiff (minimum 2 frames)
            if len(batch) < 2:
                batch = [batch[0], batch[0]]

            log.info("[M8] AnimateDiff batch %d: frames %d–%d",
                     batch_idx, i, end - 1)
            rendered = self.render_batch(batch, prompt_override)

            if progress_callback and callable(progress_callback):
                progress_callback(end, total)

            if i == 0:
                all_frames.extend(rendered)
            else:
                # Skip the first frame (overlap with previous batch)
                all_frames.extend(rendered[1:])

            batch_idx += 1
            # Advance with 1 frame overlap
            i = end - 1 if end < total else total

        return all_frames[:total]

    # ─────────────────────────────────────────────────────────────────────
    @property
    def is_ready(self) -> bool:
        """Whether the pipeline is loaded and ready."""
        return self._ready

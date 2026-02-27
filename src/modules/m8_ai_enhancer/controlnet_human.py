"""
ControlNet OpenPose human renderer  –  Physics-verified skeleton → photorealistic frames.

Architecture
~~~~~~~~~~~~
This module sits between the physics engine (M5) and the render engine (M7):

    M5  PyBullet physics step
     │
     ▼  get_link_world_positions()  →  21-joint (T, 21, 3) skeleton
     │
     ▼  SkeletonProjector  →  OpenPose-style 2-D skeleton image (512×768)
     │
     ▼  ControlNetHumanRenderer  →  photorealistic RGB frame  (512×768)
     │
     ▼  M7 RenderEngine  →  colour-grade, vignette, motion blur  →  MP4

The skeleton positions fed into the projector are the *output* of PyBullet,
NOT the raw KIT-ML data.  Gravity, ground-contact, and joint-limit forces
have already been applied, making the result physics-constrained.

VRAM
~~~~
SD 1.5 + ControlNet OpenPose in float16 + attention-slicing ≈ 3.5 GB,
compatible with RTX 3050 Laptop (4 GB VRAM).

First run downloads ~5.5 GB of model weights from HuggingFace:
    lllyasviel/sd-controlnet-openpose   (~700 MB)
    runwayml/stable-diffusion-v1-5      (~4.3 GB)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── KIT-ML 21-joint bone connectivity (same as physics_renderer.py) ──────────
BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # spine + head
    (2, 5), (5, 6), (6, 7),                   # left arm
    (2, 8), (8, 9), (9, 10),                  # right arm
    (0, 11), (0, 16), (11, 16),               # pelvis → hips
    (11, 12), (12, 13), (13, 14), (14, 15),   # left leg
    (16, 17), (17, 18), (18, 19), (19, 20),   # right leg
    (5, 8),                                    # shoulder girdle
]

# Bone-group assignments for colour coding (aligned with BONES list)
_BONE_GROUPS: List[str] = (
    ["spine"] * 4
    + ["l_arm"] * 3
    + ["r_arm"] * 3
    + ["pelvis"] * 3
    + ["l_leg"] * 4
    + ["r_leg"] * 4
    + ["girdle"] * 1
)

# OpenPose-style colours per group (RGB)
_GROUP_COLOR: dict[str, Tuple[int, int, int]] = {
    "spine":   (255, 200,   0),
    "l_arm":   ( 76, 182, 255),
    "r_arm":   (255, 130,  76),
    "pelvis":  (180, 255, 180),
    "l_leg":   (  0, 200, 120),
    "r_leg":   (160,  60, 255),
    "girdle":  (255,  80,  80),
}


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Projector  –  3-D (21, 3) → 2-D (21, 2) + OpenPose skeleton image
# ─────────────────────────────────────────────────────────────────────────────

class SkeletonProjector:
    """Project physics-verified 3-D skeleton onto a 2-D canvas.

    Uses a perspective camera at a fixed yaw/elevation relative to the
    character root.  The projection is simple pinhole – no PyBullet
    renderer involved.
    """

    def __init__(
        self,
        img_w: int = 512,
        img_h: int = 768,
        cam_yaw_deg: float = 15.0,
        joint_radius: int = 5,
        bone_thickness: int = 3,
    ) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.cam_yaw = np.radians(cam_yaw_deg)
        self.joint_radius = joint_radius
        self.bone_thickness = bone_thickness

    # ------------------------------------------------------------------
    def project(self, joints_3d: np.ndarray) -> np.ndarray:
        """Orthographic projection of (21, 3) Y-up mm coords → (21, 2) pixels.

        Returns (21, 2) int32 array of (col, row) screen coordinates.
        """
        pts = joints_3d.copy()

        # Centre body on canvas (cancel X drift and Z forward travel)
        pts[:, 0] -= pts[0, 0]
        pts[:, 2] -= pts[0, 2]

        # Yaw rotation around Y axis
        cos_y, sin_y = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        x_rot = cos_y * pts[:, 0] + sin_y * pts[:, 2]
        pts[:, 0] = x_rot

        # Screen mapping: col ← X, row ← -Y (flip so head is at top)
        screen_x = pts[:, 0]
        screen_y = pts[:, 1]

        padding = 0.12
        x_range = max(screen_x.max() - screen_x.min(), 1.0)
        y_range = max(screen_y.max() - screen_y.min(), 1.0)
        scale = min(
            self.img_w * (1 - 2 * padding) / x_range,
            self.img_h * (1 - 2 * padding) / y_range,
        )

        col = (screen_x - screen_x.min()) * scale + self.img_w * padding
        row = self.img_h - ((screen_y - screen_y.min()) * scale + self.img_h * padding)
        return np.stack([col, row], axis=-1).astype(np.int32)

    # ------------------------------------------------------------------
    def draw_skeleton(self, uv: np.ndarray) -> np.ndarray:
        """Render OpenPose-style stick figure on a black canvas.

        Parameters
        ----------
        uv : ndarray (21, 2)
            Pixel coordinates as returned by :meth:`project`.

        Returns
        -------
        ndarray (H, W, 3) uint8 RGB
        """
        canvas = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        for (j_a, j_b), group in zip(BONES, _BONE_GROUPS):
            color = _GROUP_COLOR[group]
            pt_a = (int(uv[j_a, 0]), int(uv[j_a, 1]))
            pt_b = (int(uv[j_b, 0]), int(uv[j_b, 1]))
            cv2.line(canvas, pt_a, pt_b, color, self.bone_thickness,
                     cv2.LINE_AA)

        for col_px, row_px in uv:
            pt = (int(col_px), int(row_px))
            cv2.circle(canvas, pt, self.joint_radius + 1, (60, 60, 60), -1)
            cv2.circle(canvas, pt, self.joint_radius, (255, 255, 255), -1)

        return canvas

    # ------------------------------------------------------------------
    def render(self, joints_3d: np.ndarray) -> np.ndarray:
        """Project *and* draw in one call.  Returns (H, W, 3) uint8 RGB."""
        uv = self.project(joints_3d)
        return self.draw_skeleton(uv)


# ─────────────────────────────────────────────────────────────────────────────
# ControlNet Human Renderer
# ─────────────────────────────────────────────────────────────────────────────

class ControlNetHumanRenderer:
    """Wraps SD 1.5 + ControlNet OpenPose for per-frame human rendering.

    Call :meth:`setup` once to load models, then :meth:`render_frame` per
    skeleton image.  All inference uses ``float16`` with
    ``attention_slicing(1)`` to fit within 4 GB VRAM.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/sd-controlnet-openpose",
        device: str = "cuda",
        prompt: str = (
            "a person walking, full body, studio lighting, "
            "photorealistic, 8k, cinematic, depth of field, "
            "professional photography, natural pose"
        ),
        negative_prompt: str = (
            "blurry, cartoon, sketch, deformed, extra limbs, "
            "bad anatomy, low quality, watermark, text, logo, "
            "cropped, out of frame"
        ),
        num_steps: int = 15,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.9,
        seed: int = 42,
    ) -> None:
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.device = device
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale
        self.seed = seed

        self._pipe = None
        self._ready = False

    # ------------------------------------------------------------------
    def setup(self) -> bool:
        """Load SD 1.5 + ControlNet OpenPose.  Returns True on success."""
        try:
            import torch
            from diffusers.pipelines.controlnet.pipeline_controlnet import (
                StableDiffusionControlNetPipeline,
            )
            from diffusers.models.controlnets.controlnet import ControlNetModel
            from diffusers.schedulers.scheduling_unipc_multistep import (
                UniPCMultistepScheduler,
            )

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            log.info("Loading ControlNet OpenPose …")
            controlnet = ControlNetModel.from_pretrained(
                self.controlnet_id, torch_dtype=dtype,
            )

            log.info("Loading SD 1.5 …")
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

            pipe.scheduler = UniPCMultistepScheduler.from_config(
                pipe.scheduler.config,
            )

            # VRAM optimisation for RTX 3050 (4 GB):
            # - CPU offload keeps only the active sub-model on GPU,
            #   preventing VRAM thrashing on small GPUs.
            # - Attention slicing reduces peak memory per attention block.
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
            log.info("ControlNetHumanRenderer ready on %s", self.device)
            return True

        except Exception as exc:
            log.error("ControlNetHumanRenderer setup failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    def render_frame(
        self,
        skeleton_img: np.ndarray,
        frame_seed: Optional[int] = None,
        prompt_override: Optional[str] = None,
    ) -> np.ndarray:
        """Run one ControlNet inference.

        Parameters
        ----------
        skeleton_img : ndarray (H, W, 3)
            OpenPose-style skeleton on black background.
        frame_seed : int, optional
            Per-frame seed.  Defaults to ``self.seed``.
        prompt_override : str, optional
            Override the default prompt for this frame.

        Returns
        -------
        ndarray (H, W, 3) uint8 RGB – photorealistic human frame.
        """
        if not self._ready or self._pipe is None:
            log.warning("ControlNet not loaded — returning skeleton image")
            return skeleton_img

        import torch
        from PIL import Image

        pipe = self._pipe
        pil_img = Image.fromarray(skeleton_img)
        seed = frame_seed if frame_seed is not None else self.seed
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        prompt = prompt_override or self.prompt

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                image=pil_img,
                num_inference_steps=self.num_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_scale,
                generator=generator,
            )

        result_img = output.images[0]  # type: ignore[union-attr]
        return np.array(result_img)

    # ------------------------------------------------------------------
    def render_sequence(
        self,
        skeleton_images: List[np.ndarray],
        prompt_override: Optional[str] = None,
        progress_callback: Optional[object] = None,
    ) -> List[np.ndarray]:
        """Render a full sequence of skeleton images.

        Temporal-consistency strategy
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        All frames share the **same** base noise seed so that the diffusion
        process starts from an identical latent tensor.  The only variation
        between frames is the ControlNet conditioning image (the skeleton
        pose).  This keeps background, skin tone, clothing, and lighting
        consistent across the clip — a technique documented in ControlNet
        (Zhang et al. 2023, arXiv:2302.05543) for coherent video-style
        outputs.

        Previous approach ``seed + i`` caused every frame to start from a
        completely different noise pattern, leading to severe flickering.

        Returns list of (H, W, 3) uint8 RGB frames.
        """
        if not self._ready:
            log.warning("ControlNet not loaded — returning skeleton images")
            return skeleton_images

        frames: List[np.ndarray] = []
        total = len(skeleton_images)

        for i, skel in enumerate(skeleton_images):
            # Same seed for every frame → consistent appearance
            frame = self.render_frame(
                skel,
                frame_seed=self.seed,
                prompt_override=prompt_override,
            )
            frames.append(frame)

            if progress_callback and callable(progress_callback):
                progress_callback(i + 1, total)

            if (i + 1) % 5 == 0 or i == total - 1:
                log.info("[M8] ControlNet: %d/%d frames rendered", i + 1, total)

        return frames

    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """Whether the pipeline is loaded and ready for inference."""
        return self._ready

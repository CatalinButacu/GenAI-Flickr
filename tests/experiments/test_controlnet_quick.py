"""
Quick ControlNet test — generates 2-3 photorealistic human frames
from physics-verified skeleton poses.

This script:
  1. Loads a KIT-ML walking clip
  2. Projects 3 skeleton frames → OpenPose images
  3. Runs SD 1.5 + ControlNet OpenPose → photorealistic RGB
  4. Saves side-by-side comparison: skeleton | photorealistic

Run:  python examples/test_controlnet_quick.py
Time: ~5 min on RTX 3050 (3 frames × ~90s each)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    import torch

    out_dir = "outputs/controlnet_test"
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load a walking clip from KIT-ML ─────────────────────────
    log.info("Loading KIT-ML motion data…")
    from src.modules.motion_generator import MotionGenerator

    mg = MotionGenerator(use_retrieval=True, use_ssm=False)
    clip = mg.generate("walk forward", num_frames=60)

    if clip is None or clip.raw_joints is None:
        log.error("No raw_joints — cannot proceed")
        return

    raw = clip.raw_joints  # (T, 21, 3) mm Y-up
    log.info("Loaded clip: %s, shape=%s", clip.action, raw.shape)

    # ── Step 2: Pick 3 frames and project to OpenPose ───────────────────
    from src.modules.ai_enhancer import SkeletonProjector

    projector = SkeletonProjector(img_w=512, img_h=512, cam_yaw_deg=15.0)
    indices = [0, len(raw) // 2, len(raw) - 1]
    skeleton_imgs = []

    for idx in indices:
        skel = projector.render(raw[idx])
        skeleton_imgs.append(skel)
        path = os.path.join(out_dir, f"skeleton_frame{idx:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(skel, cv2.COLOR_RGB2BGR))
        log.info("  Skeleton → %s", path)

    # ── Step 3: Load ControlNet + SD 1.5 ────────────────────────────────
    log.info("Loading SD 1.5 + ControlNet OpenPose (this may take 30-60s)…")

    from diffusers.models.controlnets.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.pipeline_controlnet import (
        StableDiffusionControlNetPipeline,
    )
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )

    dtype = torch.float16

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)

    log.info("Pipeline loaded — starting inference")

    # ── Step 4: Generate photorealistic frames ──────────────────────────
    from PIL import Image

    prompt = (
        "a person walking naturally, full body visible, "
        "photorealistic, 8k uhd, cinematic lighting, "
        "professional photography, studio backdrop, "
        "detailed clothing, natural skin texture, "
        "shallow depth of field"
    )
    negative = (
        "blurry, cartoon, sketch, anime, deformed, extra limbs, "
        "bad anatomy, low quality, watermark, text, logo, "
        "cropped, out of frame, nude, nsfw, disfigured"
    )

    seed = 42

    results = []
    for i, skel_img in enumerate(skeleton_imgs):
        t0 = time.time()
        log.info("  Generating frame %d/%d …", i + 1, len(skeleton_imgs))

        pil_skel = Image.fromarray(skel_img)

        # Reset generator to same seed for consistency
        generator = torch.Generator(device="cpu").manual_seed(seed)

        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=pil_skel,
                num_inference_steps=20,
                guidance_scale=7.5,
                controlnet_conditioning_scale=1.0,
                generator=generator,
            )

        result_img = np.array(output.images[0])  # type: ignore[union-attr]
        results.append(result_img)

        elapsed = time.time() - t0
        log.info("  Frame %d done in %.1fs", i + 1, elapsed)

        # Save individual result
        path = os.path.join(out_dir, f"photorealistic_frame{indices[i]:04d}.png")
        cv2.imwrite(path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        log.info("  Saved → %s", path)

    # ── Step 5: Create side-by-side comparison ──────────────────────────
    log.info("Creating comparison grid…")

    h, w = 512, 512
    n = len(results)
    grid = np.zeros((h * 2, w * n, 3), dtype=np.uint8)

    for i in range(n):
        # Top row: skeleton input
        skel_resized = cv2.resize(skeleton_imgs[i], (w, h))
        grid[0:h, i*w:(i+1)*w] = skel_resized
        # Bottom row: photorealistic output
        result_resized = cv2.resize(results[i], (w, h))
        grid[h:2*h, i*w:(i+1)*w] = result_resized

    # Add labels
    cv2.putText(grid, "SKELETON INPUT (OpenPose)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(grid, "CONTROLNET OUTPUT (SD 1.5)", (10, h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for i, idx in enumerate(indices):
        cv2.putText(grid, f"Frame {idx}", (i*w + 10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    grid_path = os.path.join(out_dir, "comparison_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    log.info("Comparison grid → %s", grid_path)

    # ── Step 6: Open the result ─────────────────────────────────────────
    import subprocess
    abs_path = os.path.abspath(grid_path)
    subprocess.Popen(["cmd", "/c", "start", "", abs_path])
    print(f"\n{'━' * 60}")
    print(f"  Comparison grid saved → {abs_path}")
    print(f"  Individual frames in → {os.path.abspath(out_dir)}/")
    print(f"{'━' * 60}")

    # Clean up GPU memory
    del pipe, controlnet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

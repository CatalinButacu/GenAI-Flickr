"""
Minimal 1-frame ControlNet test — proves photorealistic output works.
Generates ONE frame with 15 steps (lighter) to fit within time limits.

Run:  python examples/test_controlnet_1frame.py
Time: ~8-10 min on RTX 3050 4GB
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

    # ── Load one walking pose from KIT-ML ───────────────────────────────
    log.info("Loading KIT-ML motion…")
    from src.modules.m4_motion_generator import MotionGenerator

    mg = MotionGenerator(use_retrieval=True, use_ssm=False)
    clip = mg.generate("walk forward", num_frames=60)
    if clip is None or clip.raw_joints is None:
        log.error("No raw_joints")
        return

    raw = clip.raw_joints
    mid_frame = raw[len(raw) // 2]  # middle frame — mid-stride
    log.info("Using frame %d of %d, shape=%s", len(raw) // 2, len(raw), mid_frame.shape)

    # ── Project to OpenPose ─────────────────────────────────────────────
    from src.modules.m8_ai_enhancer import SkeletonProjector

    projector = SkeletonProjector(img_w=512, img_h=512, cam_yaw_deg=15.0)
    skel_img = projector.render(mid_frame)

    # Save skeleton UV coordinates for later verification
    physics_uv = projector.project(mid_frame)
    uv_path = os.path.join(out_dir, "physics_uv.npy")
    np.save(uv_path, physics_uv)
    log.info("Saved physics UV coords → %s  (shape %s)", uv_path, physics_uv.shape)

    skel_path = os.path.join(out_dir, "input_skeleton.png")
    cv2.imwrite(skel_path, cv2.cvtColor(skel_img, cv2.COLOR_RGB2BGR))
    log.info("Skeleton input → %s", skel_path)

    # ── Load SD 1.5 + ControlNet ────────────────────────────────────────
    log.info("Loading models… (using cached weights, ~20-30s)")
    t_load = time.time()

    from diffusers.models.controlnets.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.pipeline_controlnet import (
        StableDiffusionControlNetPipeline,
    )
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing(1)

    log.info("Models loaded in %.0fs", time.time() - t_load)

    # ── Generate 1 photorealistic frame ─────────────────────────────────
    from PIL import Image

    prompt = (
        "a person walking naturally, full body, "
        "photorealistic, 8k uhd, cinematic studio lighting, "
        "detailed clothing, natural skin texture, "
        "shallow depth of field, professional photography"
    )
    negative = (
        "blurry, cartoon, sketch, anime, deformed, extra limbs, "
        "bad anatomy, low quality, watermark, text, disfigured, "
        "nude, nsfw, cropped, out of frame"
    )

    log.info("Starting ControlNet inference (15 steps)…")
    t_infer = time.time()

    generator = torch.Generator(device="cpu").manual_seed(42)

    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=Image.fromarray(skel_img),
            num_inference_steps=15,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=generator,
        )

    result = np.array(output.images[0])  # type: ignore[union-attr]
    elapsed = time.time() - t_infer
    log.info("Inference done in %.0fs (%.1fs/step)", elapsed, elapsed / 15)

    # ── Save result ─────────────────────────────────────────────────────
    result_path = os.path.join(out_dir, "photorealistic_output.png")
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    log.info("Photorealistic output → %s", result_path)

    # ── Side-by-side comparison ─────────────────────────────────────────
    h, w = 512, 512
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = cv2.resize(skel_img, (w, h))
    comparison[:, w:] = cv2.resize(result, (w, h))

    # Labels
    cv2.putText(comparison, "SKELETON INPUT", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(comparison, "CONTROLNET OUTPUT", (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    comp_path = os.path.join(out_dir, "comparison_1frame.png")
    cv2.imwrite(comp_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    log.info("Comparison → %s", comp_path)

    # Open it
    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", os.path.abspath(comp_path)])

    print(f"\n{'━' * 50}")
    print(f"  Skeleton   → {os.path.abspath(skel_path)}")
    print(f"  Output     → {os.path.abspath(result_path)}")
    print(f"  Comparison → {os.path.abspath(comp_path)}")
    print(f"  Time       → {elapsed:.0f}s ({elapsed/15:.1f}s/step)")
    print(f"{'━' * 50}")

    del pipe, controlnet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

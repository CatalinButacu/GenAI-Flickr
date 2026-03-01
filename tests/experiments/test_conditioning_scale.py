"""
Conditioning scale ablation — measures physics adherence at different
ControlNet conditioning scales.

Generates one frame per scale value and runs the adherence verifier
on each.  Produces a comparison grid and CSV of metrics.

Run: python examples/test_conditioning_scale.py
Time: ~10 min per scale × N scales
"""
from __future__ import annotations

import csv
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

# Scales to test (ascending: loose → tight conditioning)
SCALES = [0.7, 1.0, 1.3, 1.6, 2.0]


def main() -> None:
    import torch
    from PIL import Image
    from diffusers.models.controlnets.controlnet import ControlNetModel
    from diffusers.pipelines.controlnet.pipeline_controlnet import (
        StableDiffusionControlNetPipeline,
    )
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )
    from src.modules.motion_generator import MotionGenerator
    from src.modules.physics_engine.physics_verifier import (
        PhysicsAdherenceVerifier,
    )
    from src.modules.ai_enhancer import SkeletonProjector

    out_dir = "outputs/scale_ablation"
    os.makedirs(out_dir, exist_ok=True)

    # ── Get one skeleton pose ───────────────────────────────────────────
    log.info("Loading motion…")
    mg = MotionGenerator(use_retrieval=True, use_ssm=False)
    clip = mg.generate("walk forward", num_frames=60)
    if clip is None or clip.raw_joints is None:
        log.error("No raw_joints"); return

    mid_frame = clip.raw_joints[len(clip.raw_joints) // 2]
    projector = SkeletonProjector(img_w=512, img_h=512, cam_yaw_deg=15.0)
    skel_img = projector.render(mid_frame)
    physics_uv = projector.project(mid_frame)
    np.save(os.path.join(out_dir, "physics_uv.npy"), physics_uv)

    cv2.imwrite(
        os.path.join(out_dir, "skeleton_input.png"),
        cv2.cvtColor(skel_img, cv2.COLOR_RGB2BGR),
    )

    # ── Load models once ────────────────────────────────────────────────
    log.info("Loading SD 1.5 + ControlNet…")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16,
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
    log.info("Models loaded.")

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

    # ── Verifier ────────────────────────────────────────────────────────
    verifier = PhysicsAdherenceVerifier(threshold_px=30.0, img_size=(512, 512))
    if not verifier.setup():
        log.error("MediaPipe failed"); return

    # ── Run ablation ────────────────────────────────────────────────────
    results_csv = []
    frames = []

    for scale in SCALES:
        log.info("=" * 50)
        log.info("Generating with conditioning_scale=%.1f", scale)
        t0 = time.time()

        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=Image.fromarray(skel_img),
                num_inference_steps=15,
                guidance_scale=7.5,
                controlnet_conditioning_scale=scale,
                generator=generator,
            )

        frame_rgb = np.array(output.images[0])  # type: ignore[union-attr]
        elapsed = time.time() - t0
        log.info("Done in %.0fs", elapsed)

        # Save frame
        fname = f"scale_{scale:.1f}.png"
        cv2.imwrite(
            os.path.join(out_dir, fname),
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        )

        # Verify
        vr = verifier.verify_frame(frame_rgb, physics_uv, frame_index=0)
        log.info(
            "  scale=%.1f  raw=%.1f%%  aligned=%.1f%%  err=%.1fpx  "
            "offset=%.0fpx  scale_ratio=%.2f",
            scale, vr.adherence_score, vr.aligned_adherence_score,
            vr.mean_error_px, vr.offset_px, vr.scale_ratio,
        )

        results_csv.append({
            "conditioning_scale": scale,
            "raw_adherence_pct": round(vr.adherence_score, 1),
            "aligned_adherence_pct": round(vr.aligned_adherence_score, 1),
            "mean_error_px": round(vr.mean_error_px, 1),
            "aligned_mean_error_px": round(vr.aligned_mean_error_px, 1),
            "offset_px": round(vr.offset_px, 1),
            "scale_ratio": round(vr.scale_ratio, 2),
            "detected_joints": vr.detected_joints,
            "time_s": round(elapsed, 0),
        })
        frames.append((scale, frame_rgb))

    verifier.cleanup()

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "scale_ablation.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_csv[0].keys()))
        writer.writeheader()
        writer.writerows(results_csv)
    log.info("CSV → %s", csv_path)

    # ── Comparison grid ─────────────────────────────────────────────────
    n = len(frames) + 1  # skeleton + N scales
    cell = 256
    grid = np.zeros((cell, cell * n, 3), dtype=np.uint8)

    # First cell: skeleton
    grid[:, :cell] = cv2.resize(skel_img, (cell, cell))
    cv2.putText(grid, "Skeleton", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, (scale, frame_rgb) in enumerate(frames):
        x0 = (i + 1) * cell
        grid[:, x0:x0 + cell] = cv2.resize(frame_rgb, (cell, cell))
        label = f"s={scale:.1f}"
        adh = results_csv[i]["aligned_adherence_pct"]
        cv2.putText(grid, label, (x0 + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(grid, f"{adh}%", (x0 + 5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    grid_path = os.path.join(out_dir, "scale_comparison.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    log.info("Grid → %s", grid_path)

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'━' * 75}")
    print("  CONDITIONING SCALE ABLATION \u2014 Physics Adherence")
    print(f"{'━' * 75}")
    print(f"  {'Scale':>6s}  {'Raw%':>5s}  {'Aligned%':>8s}  "
          f"{'Error':>6s}  {'AlignErr':>8s}  {'Offset':>6s}  {'ScaleR':>6s}")
    print(f"  {'─' * 6}  {'─' * 5}  {'─' * 8}  "
          f"{'─' * 6}  {'─' * 8}  {'─' * 6}  {'─' * 6}")
    for r in results_csv:
        print(f"  {r['conditioning_scale']:6.1f}  {r['raw_adherence_pct']:5.1f}  "
              f"{r['aligned_adherence_pct']:8.1f}  {r['mean_error_px']:6.1f}  "
              f"{r['aligned_mean_error_px']:8.1f}  {r['offset_px']:6.1f}  "
              f"{r['scale_ratio']:6.2f}")
    print(f"{'━' * 75}")

    del pipe, controlnet
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

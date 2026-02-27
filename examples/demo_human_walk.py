"""Demo: KIT-ML walk motion → OpenPose skeleton → ControlNet → ultra-realistic MP4.

Pipeline
--------
1. Find a "walk forward" clip in the KIT-ML dataset (M4 data)
2. Project 3-D joint positions to a 2-D camera view
3. Render each frame as an OpenPose-style skeleton image
4. Feed skeleton frames into ControlNet OpenPose + SD 1.5 (M8)
5. Apply M7 post-processing (motion blur, color grade, vignette)
6. Write final MP4

Requirements: diffusers, transformers, torch (already in requirements.txt)
First run will download ~5.5 GB of model weights (SD 1.5 + ControlNet OpenPose).
VRAM: 3.5–4 GB  (float16 + attention_slicing)

Run:
    python examples/demo_human_walk.py
    python examples/demo_human_walk.py --id 00964 --steps 25 --frames 32
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ── make src importable when run directly ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── KIT-ML skeleton: 21 joint indices ──────────────────────────────────────
#
#  KIT-ML uses Y-up millimetres.  Connectivity derived from coordinate analysis.
#
#   0=pelvis   1=low_spine  2=chest    3=neck     4=head
#   5=L.shld   6=L.elbow    7=L.wrist
#   8=R.shld   9=R.elbow   10=R.wrist
#  11=L.hip   12=L.knee   13=L.ankle  14=L.foot  15=L.toe
#  16=R.hip   17=R.knee   18=R.ankle  19=R.foot  20=R.toe
#
_BONES = [
    # spine & head
    (0, 1), (1, 2), (2, 3), (3, 4),
    # left arm
    (2, 5), (5, 6), (6, 7),
    # right arm
    (2, 8), (8, 9), (9, 10),
    # pelvis → hips
    (0, 11), (0, 16), (11, 16),
    # left leg
    (11, 12), (12, 13), (13, 14), (14, 15),
    # right leg
    (16, 17), (17, 18), (18, 19), (19, 20),
    # shoulder girdle
    (5, 8),
]

# OpenPose-style colours per bone group (BGR → converted to RGB in render)
_BONE_COLORS_RGB = {
    "spine":   (255, 200, 0),    # gold
    "l_arm":   (76, 182, 255),   # sky blue
    "r_arm":   (255, 130, 76),   # orange
    "pelvis":  (180, 255, 180),  # light green
    "l_leg":   (0,  200, 120),   # green
    "r_leg":   (160, 60, 255),   # purple
    "girdle":  (255, 80,  80),   # red
}
_BONE_GROUP = (
    ["spine"]  * 4 +
    ["l_arm"]  * 3 +
    ["r_arm"]  * 3 +
    ["pelvis"] * 3 +
    ["l_leg"]  * 4 +
    ["r_leg"]  * 4 +
    ["girdle"] * 1
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_preferred_clip(
    joints_dir: Path, texts_dir: Path, preferred_id: str, query: str
) -> Optional[tuple[np.ndarray, str, str]]:
    """Try to load a specific clip by ID; return None if not found."""
    jpath = joints_dir / f"{preferred_id}.npy"
    if not jpath.exists():
        return None
    tpath = texts_dir / f"{preferred_id}.txt"
    text = tpath.read_text().split("#")[0].strip() if tpath.exists() else query
    joints = np.load(jpath)
    log.info("Using specified clip %s — '%s'  shape=%s", preferred_id, text, joints.shape)
    return joints, preferred_id, text


def _search_clips(
    joints_dir: Path, texts_dir: Path, query: str
) -> list[tuple[str, str, np.ndarray]]:
    """Search KIT-ML for clips whose description contains *query*."""
    candidates = []
    for tpath in sorted(texts_dir.glob("*.txt")):
        raw = tpath.read_text().split("#")[0].strip().lower()
        if query not in raw:
            continue
        jpath = joints_dir / (tpath.stem + ".npy")
        if not jpath.exists():
            continue
        j = np.load(jpath)
        # Prefer clips of moderate length (40–80 frames ≈ 2–4 s)
        if 30 <= len(j) <= 100:
            candidates.append((tpath.stem, raw, j))
    return candidates


def find_walk_clip(kit_dir: str = "data/KIT-ML",
                   query: str = "walk",
                   preferred_id: Optional[str] = None) -> tuple[np.ndarray, str, str]:
    """Return (joints, sample_id, text) for the best walking clip found."""
    kit = Path(kit_dir)
    texts_dir  = kit / "texts"
    joints_dir = kit / "new_joints"

    if preferred_id:
        result = _load_preferred_clip(joints_dir, texts_dir, preferred_id, query)
        if result is not None:
            return result

    candidates = _search_clips(joints_dir, texts_dir, query)
    if not candidates:
        raise FileNotFoundError(
            f"No KIT-ML clip found containing '{query}'. "
            f"Looked in {texts_dir}"
        )

    # Pick the longest clip in range
    sample_id, text, joints = max(candidates, key=lambda x: len(x[2]))
    log.info("Selected clip %s — '%s'  frames=%d  (%.1fs at 20fps)",
             sample_id, text, len(joints), len(joints) / 20)
    return joints, sample_id, text


# ─────────────────────────────────────────────────────────────────────────────
# 2. Skeleton projection → 2-D skeleton image
# ─────────────────────────────────────────────────────────────────────────────

def project_joints(joints_3d: np.ndarray,
                   img_w: int = 512, img_h: int = 512,
                   cam_yaw_deg: float = 0.0) -> np.ndarray:
    """
    Orthographic projection of (21, 3) joint positions onto a 2-D canvas.

    We face the human from a yaw-rotated camera so that the walk looks natural.
    Returns (21, 2) array of (col, row) pixel coordinates.
    """
    pts = joints_3d.copy()        # (21, 3) — X, Y, Z  (Y=up, Z=forward)

    # Centre the body over the canvas (remove forward drift)
    pts[:, 0] -= pts[0, 0]        # subtract root X
    pts[:, 2] -= pts[0, 2]        # subtract root Z (cancel forward travel)

    # Apply yaw rotation around Y axis so we see the person from a mild angle
    yaw = np.radians(cam_yaw_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    x_rot =  cos_y * pts[:, 0] + sin_y * pts[:, 2]
    z_rot = -sin_y * pts[:, 0] + cos_y * pts[:, 2]
    pts[:, 0] = x_rot
    pts[:, 2] = z_rot

    # Screen: col ← X,  row ← -Y (flip so head is at top)
    screen_x = pts[:, 0]
    screen_y = pts[:, 1]    # Y-up → row decreases upward

    # Normalise to canvas with some padding
    padding = 0.12
    x_range = screen_x.max() - screen_x.min() or 1.0
    y_range = screen_y.max() - screen_y.min() or 1.0
    scale = min(img_w * (1 - 2 * padding) / x_range,
                img_h * (1 - 2 * padding) / y_range)

    col = (screen_x - screen_x.min()) * scale + img_w * padding
    row = img_h - ((screen_y - screen_y.min()) * scale + img_h * padding)

    return np.stack([col, row], axis=-1).astype(np.int32)


def render_skeleton_image(uv: np.ndarray,
                           img_w: int = 512, img_h: int = 512,
                           joint_radius: int = 5,
                           bone_thickness: int = 3) -> np.ndarray:
    """
    Draw OpenPose-style stick figure on a black canvas.
    Returns uint8 RGB array (H, W, 3).
    """
    import cv2

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # Draw bones
    for (j_a, j_b), group in zip(_BONES, _BONE_GROUP):
        color = _BONE_COLORS_RGB[group]
        pt_a = (int(uv[j_a, 0]), int(uv[j_a, 1]))
        pt_b = (int(uv[j_b, 0]), int(uv[j_b, 1]))
        cv2.line(canvas, pt_a, pt_b, color, bone_thickness, cv2.LINE_AA)

    # Draw joints (white circles with colored outlines)
    for i, (col, row) in enumerate(uv):
        pt = (int(col), int(row))
        cv2.circle(canvas, pt, joint_radius + 1, (60, 60, 60), -1)
        cv2.circle(canvas, pt, joint_radius, (255, 255, 255), -1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 3. ControlNet OpenPose enhancement
# ─────────────────────────────────────────────────────────────────────────────

def load_controlnet_pipeline(device: str = "cuda"):
    """Load SD 1.5 + ControlNet OpenPose (downloads ~5.5 GB on first run)."""
    import torch
    from diffusers.pipelines.controlnet.pipeline_controlnet import (
        StableDiffusionControlNetPipeline,
    )
    from diffusers.models.controlnets.controlnet import ControlNetModel
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler,
    )

    log.info("Loading ControlNet OpenPose model (~5.5 GB on first download) …")
    dtype = torch.float16 if device == "cuda" else torch.float32

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=dtype,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # VRAM safety for RTX 3050 (4 GB)
    pipe.enable_attention_slicing(1)
    try:
        pipe.enable_xformers_memory_efficient_attention()
        log.info("xformers attention enabled")
    except Exception:
        pass

    log.info("ControlNet pipeline ready on %s", device)
    return pipe


def enhance_frame(pipe, skeleton_img: np.ndarray,
                  prompt: str,
                  negative_prompt: str,
                  num_steps: int,
                  guidance_scale: float,
                  controlnet_scale: float,
                  seed: int) -> np.ndarray:
    """Run one ControlNet inference step. Returns uint8 RGB (H, W, 3)."""
    import torch
    from PIL import Image

    pil_skeleton = Image.fromarray(skeleton_img)
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pil_skeleton,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        ).images[0]

    return np.array(result)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Assemble & export video
# ─────────────────────────────────────────────────────────────────────────────

def write_video(frames: list[np.ndarray], path: str, fps: int) -> None:
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=16)
    for f in frames:
        writer.append_data(f)
    writer.close()
    log.info("Saved → %s  (%d frames @ %dfps)", path, len(frames), fps)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ultra-realistic human walk demo")
    parser.add_argument("--id", default=None, help="Force a specific KIT-ML sample ID (e.g. 00964)")
    parser.add_argument("--query", default="walk", help="Motion keyword to search in KIT-ML")
    parser.add_argument("--frames", type=int, default=24,
                        help="Number of frames to render (default 24 = 1.2s video)")
    parser.add_argument("--steps", type=int, default=20,
                        help="ControlNet inference steps per frame (default 20, higher = better quality)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width",  type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--yaw", type=float, default=15.0,
                        help="Camera yaw angle in degrees (0=front, 90=side)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--prompt", default=(
        "a person walking forward on a park path, golden hour, photorealistic, "
        "cinematic lighting, 8k, professional photography, depth of field, ultra detailed"
    ))
    parser.add_argument("--negative", default=(
        "blurry, cartoon, sketch, deformed, extra limbs, bad anatomy, "
        "low quality, watermark, text, logo"
    ))
    parser.add_argument("--controlnet-scale", type=float, default=0.9,
                        help="ControlNet conditioning strength (0.8–1.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (incremented per frame for variety)")
    parser.add_argument("--skeleton-only", action="store_true",
                        help="Export only the skeleton animation, skip ControlNet")
    parser.add_argument("--kit-dir", default="data/KIT-ML")
    args = parser.parse_args()

    out_dir = "outputs/videos"

    # ── step 1: load motion data ────────────────────────────────────────────
    joints_all, sample_id, text = find_walk_clip(
        kit_dir=args.kit_dir, query=args.query, preferred_id=args.id
    )
    log.info("Motion: '%s' — %d raw frames", text, len(joints_all))

    # Subsample / trim to requested frame count
    total = len(joints_all)
    indices = np.linspace(0, total - 1, min(args.frames, total), dtype=int)
    joints_seq = joints_all[indices]   # (F, 21, 3)
    log.info("Using %d frames (subsampled from %d)", len(joints_seq), total)

    # ── step 2: project & render skeletons ─────────────────────────────────
    log.info("Rendering skeleton frames …")
    skeleton_frames: list[np.ndarray] = []
    for t, frame_joints in enumerate(joints_seq):
        uv = project_joints(frame_joints, img_w=args.width, img_h=args.height, cam_yaw_deg=args.yaw)
        skel = render_skeleton_image(uv, img_w=args.width, img_h=args.height)
        skeleton_frames.append(skel)

    # Save skeleton-only preview
    skel_path = os.path.join(out_dir, f"skeleton_{sample_id}.mp4")
    write_video(skeleton_frames, skel_path, fps=args.fps)
    log.info("Skeleton preview → %s", skel_path)

    if args.skeleton_only:
        log.info("--skeleton-only flag set, stopping here.")
        return

    # ── step 3: ControlNet enhancement ──────────────────────────────────────
    log.info("Loading ControlNet pipeline …")
    pipe = load_controlnet_pipeline(args.device)

    enhanced_frames: list[np.ndarray] = []
    total_frames = len(skeleton_frames)
    for i, skel in enumerate(skeleton_frames):
        log.info("  [%d/%d] ControlNet inference (steps=%d) …", i + 1, total_frames, args.steps)
        img = enhance_frame(
            pipe, skel,
            prompt=args.prompt,
            negative_prompt=args.negative,
            num_steps=args.steps,
            guidance_scale=7.5,
            controlnet_scale=args.controlnet_scale,
            seed=args.seed + i,
        )
        enhanced_frames.append(img)

    # ── step 4: M7 post-processing ──────────────────────────────────────────
    log.info("Applying M7 cinematic post-processing …")
    from src.modules.m7_render_engine import RenderEngine, RenderSettings

    class _WrappedFrame:
        """Wrap a plain numpy array to look like FrameData for RenderEngine."""
        def __init__(self, rgb: np.ndarray):
            self.rgb = rgb

    settings = RenderSettings(
        motion_blur=True,
        motion_blur_alpha=0.7,
        dof=False,           # no depth buffer for ControlNet output
        color_grade=True,
        saturation=1.3,
        contrast=1.1,
        gamma=0.90,
        tint=(1.03, 1.0, 0.95),
        vignette=True,
        vignette_strength=0.5,
        film_grain=True,
        grain_sigma=3.0,
    )
    engine = RenderEngine(settings)
    wrapped = [_WrappedFrame(f) for f in enhanced_frames]
    processed = engine._process_frames(wrapped)   # noqa: SLF001 — internal but acceptable for demo

    # ── step 5: write final video ────────────────────────────────────────────
    final_path = os.path.join(out_dir, f"human_walk_{sample_id}_enhanced.mp4")
    write_video(processed, final_path, fps=args.fps)

    # Comparison side-by-side (skeleton | enhanced)
    try:
        import cv2
        from PIL import Image
        side_frames = []
        for skel, enh in zip(skeleton_frames, processed):
            skel_resized = np.array(Image.fromarray(skel).resize((args.width // 2, args.height // 2)))
            enh_resized  = np.array(Image.fromarray(enh).resize((args.width // 2, args.height // 2)))
            side_frames.append(np.concatenate([skel_resized, enh_resized], axis=1))
        comparison_path = os.path.join(out_dir, f"comparison_{sample_id}.mp4")
        write_video(side_frames, comparison_path, fps=args.fps)
        log.info("Comparison video → %s", comparison_path)
    except Exception as e:
        log.warning("Comparison video skipped: %s", e)

    import subprocess
    subprocess.Popen(["cmd", "/c", "start", "", final_path])   # open in default player
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  DONE \u2014 ultra-realistic walk demo complete!")
    print(f"  Motion clip  :  {sample_id} — '{text}'")
    print(f"  Frames       :  {len(processed)}")
    print(f"  Final video  :  {final_path}")
    print(f"  Skeleton     :  {skel_path}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()

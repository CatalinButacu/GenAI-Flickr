"""Pipeline orchestrator — delegates to four stage classes."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.shared.mem_profile import tracemalloc_snapshot
from src.shared.constants import DEFAULT_FPS, DEFAULT_DURATION
from src.modules.assets import ModelGenerator
from src.stages.understanding import UnderstandingStage
from src.stages.motion import MotionStage
from src.stages.physics import PhysicsStage, PhysicsResult
from src.stages.rendering import RenderingStage

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    assets_dir: str = "assets/objects"
    use_t5_parser: bool = True           # M1: prefer T5 model over rules parser
    use_asset_generation: bool = False   # M3: Shap-E (GPU)
    use_motion_generation: bool = True   # M4: SSM / AMASS retrieval
    use_physics_ssm: bool = True         # M4: PhysicsSSM refinement pass
    use_diffusion: bool = False          # M7: ControlNet / AnimateDiff (GPU)
    use_render_engine: bool = True       # M6: motion blur, DoF, color grade
    use_silhouette: bool = True          # M6: filled body silhouettes
    validate_motion: bool = True         # Biomechanical validation + repair
    max_repairs: int = 3
    fps: int = DEFAULT_FPS
    duration: float = DEFAULT_DURATION
    device: str = "cuda"
    fixed_camera: bool = False
    print_module_io: bool = True


class Pipeline:
    """Text prompt -> MP4 video.

    Delegates to four stages:
      M1+M2: UnderstandingStage  (parse + plan)
      M4:    MotionStage         (generate + refine + validate)
      M5:    PhysicsStage        (simulate)
      M6+M7: RenderingStage     (render + export)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.is_setup = False

        # Stages (created in setup)
        self.understanding: UnderstandingStage | None = None
        self.motion: MotionStage | None = None
        self.physics: PhysicsStage | None = None
        self.rendering: RenderingStage | None = None
        self.asset_gen: ModelGenerator | None = None

    def setup(self) -> None:
        t0 = time.time()
        cfg = self.config
        log.info("=" * 64)
        log.info("PIPELINE SETUP — %s", cfg.__dict__)
        log.info("=" * 64)

        self.understanding = UnderstandingStage(
            use_t5=cfg.use_t5_parser, device=cfg.device,
        )
        self.understanding.setup()

        if cfg.use_asset_generation:
            self.asset_gen = ModelGenerator(device=cfg.device)
            if not self.asset_gen.setup():
                raise RuntimeError("ModelGenerator.setup() failed")
            log.info("[M3] ModelGenerator ready")

        if cfg.use_motion_generation:
            self.motion = MotionStage(
                use_physics_ssm=cfg.use_physics_ssm,
                validate_motion=cfg.validate_motion,
                max_repairs=cfg.max_repairs,
                duration=cfg.duration,
                device=cfg.device,
            )
            self.motion.setup()

        self.physics = PhysicsStage(
            fps=cfg.fps, duration=cfg.duration,
            fixed_camera=cfg.fixed_camera,
        )

        self.rendering = RenderingStage(
            fps=cfg.fps,
            use_render_engine=cfg.use_render_engine,
            use_diffusion=cfg.use_diffusion,
            use_silhouette=cfg.use_silhouette,
            device=cfg.device,
        )
        if cfg.use_diffusion:
            self.rendering.setup()

        self.is_setup = True
        log.info("PIPELINE SETUP COMPLETE in %.2fs", time.time() - t0)

    def run(self, prompt: str, output_name: str = "output") -> dict[str, Any]:
        if not self.is_setup:
            self.setup()

        prompt = self._sanitize_prompt(prompt)
        if prompt is None:
            return {"prompt": "", "error": "empty prompt"}

        run_t0 = time.time()
        log.info("=" * 64)
        log.info("PIPELINE RUN — prompt: %r", prompt)
        log.info("=" * 64)

        # M1 + M2: Parse and plan
        parsed, planned = self.understanding.run(prompt)

        # M4: Generate motion
        with tracemalloc_snapshot("M4 motion"):
            motion_clips = (self.motion.run(parsed)
                            if self.motion else {})

        # M5: Physics simulation
        with tracemalloc_snapshot("M5 physics"):
            phys = self.physics.run(planned, motion_clips, parsed)

        # M6: Render skeleton -> RGB frames (if humanoid sim produced poses)
        if phys.needs_rendering:
            frames = self.rendering.render_frames(
                phys.skeleton_positions, phys.cam,
                action_label=phys.action_label,
                body_params=phys.body_params,
                smplx_motion=phys.smplx_motion,
                smplx_betas=phys.smplx_betas,
            )
        else:
            frames = phys.frames

        # M6: Export video
        try:
            with tracemalloc_snapshot("M6 export"):
                video_path = self.rendering.export_video(
                    phys.sim, frames, output_name,
                    output_dir=self.config.output_dir,
                )
        finally:
            phys.scene.cleanup()

        elapsed = time.time() - run_t0
        log.info("=" * 64)
        log.info("PIPELINE COMPLETE in %.2fs — video: %s", elapsed, video_path)
        log.info("=" * 64)

        return {
            "prompt": prompt,
            "parsed_scene": parsed,
            "planned_scene": planned,
            "motion_clips": motion_clips,
            "physics_frames": frames,
            "video_path": video_path,
            "elapsed_seconds": elapsed,
        }

    # ── Prompt limits ────────────────────────────────────────────────
    # T5-small tokenizer has a 512-token input limit (~1500 English chars).
    # We allow up to 2000 chars to be generous with whitespace / punctuation,
    # and the tokenizer's own truncation=True handles the actual token cap.
    # The rules-based PromptParser has no inherent length limit.
    PROMPT_MAX_CHARS: int = 2000

    @staticmethod
    def _sanitize_prompt(prompt: str) -> str | None:
        prompt = (prompt or "").strip()
        if not prompt:
            log.warning("Pipeline.run() called with empty prompt")
            return None
        if len(prompt) > Pipeline.PROMPT_MAX_CHARS:
            log.warning("Prompt truncated from %d to %d chars",
                        len(prompt), Pipeline.PROMPT_MAX_CHARS)
            prompt = prompt[:Pipeline.PROMPT_MAX_CHARS]
        return prompt

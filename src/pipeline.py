from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    assets_dir: str = "assets/objects"
    use_asset_generation: bool = False  # M3: Shap-E (GPU)
    use_motion_generation: bool = True   # M4: SSM / KIT-ML
    use_rl_controller: bool = False      # M6: PPO stub
    use_ai_enhancement: bool = False     # M8: ControlNet (GPU)
    fps: int = 24
    duration: float = 5.0
    device: str = "cuda"


class Pipeline:
    """Text prompt → MP4 video. Loads modules lazily on first run()."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._is_setup = False
        self._parser = self._planner = self._asset_gen = None
        self._motion_gen = self._enhancer = None

    def setup(self) -> None:
        from src.modules.m1_scene_understanding.prompt_parser import PromptParser
        from src.modules.m2_scene_planner import ScenePlanner

        self._parser = PromptParser()
        self._planner = ScenePlanner()
        log.info("[M1] PromptParser ready | [M2] ScenePlanner ready")

        _TOGGLE = {
            "M3": "use_asset_generation",
            "M4": "use_motion_generation",
            "M8": "use_ai_enhancement",
        }
        for tag, fn in [("M3", self._init_asset_gen), ("M4", self._init_motion_gen), ("M8", self._init_enhancer)]:
            if getattr(self.config, _TOGGLE[tag]):
                try:
                    fn()
                    log.info("[%s] ready", tag)
                except Exception as exc:
                    log.warning("[%s] disabled: %s", tag, exc)
                    setattr(self.config, _TOGGLE[tag], False)

        log.info("[M5] PhysicsEngine ready | [M6] RLController stub | [M7] RenderEngine stub")
        self._is_setup = True

    def _init_asset_gen(self) -> None:
        from src.modules.m3_asset_generator import ModelGenerator
        self._asset_gen = ModelGenerator(device=self.config.device)
        if not self._asset_gen.setup():
            raise RuntimeError("setup() returned False")

    def _init_motion_gen(self) -> None:
        from src.modules.m4_motion_generator import MotionGenerator
        self._motion_gen = MotionGenerator(backend="retrieval")
        self._motion_gen.setup()

    def _init_enhancer(self) -> None:
        from src.modules.m8_ai_enhancer import VideoRenderer
        self._enhancer = VideoRenderer(device=self.config.device)
        if not self._enhancer.setup():
            raise RuntimeError("setup() returned False")

    def run(self, prompt: str, output_name: str = "output") -> Dict[str, Any]:
        if not self._is_setup:
            self.setup()

        parsed = self._parser.parse(prompt)
        log.info("[M1] %d entities, %d actions", len(parsed.entities), len(parsed.actions))

        planned = self._planner.plan(parsed)
        log.info("[M2] %d entities positioned", len(planned.entities))

        motion_clips = self._generate_motion(parsed)
        frames, sim, scene = self._run_physics(planned)
        log.info("[M5] %d frames captured", len(frames))

        video_path = self._export_video(sim, frames, output_name)
        scene.cleanup()

        if self.config.use_ai_enhancement and self._enhancer:
            log.info("[M8] per-frame enhancement pending")

        return {
            "prompt": prompt,
            "parsed_scene": parsed,
            "planned_scene": planned,
            "motion_clips": motion_clips,
            "physics_frames": frames,
            "video_path": video_path,
        }

    def _generate_motion(self, parsed) -> Dict[str, Any]:
        if not (self.config.use_motion_generation and self._motion_gen):
            return {}
        from src.shared.vocabulary import ACTIONS
        return {
            action.actor: self._motion_gen.generate(
                ACTIONS[action.action_type].motion_clip.replace("_", " "),
                duration=self.config.duration,
                fps=self.config.fps,
            )
            for action in parsed.actions
            if (act_def := ACTIONS.get(action.action_type)) and act_def.motion_clip
        }

    def _run_physics(self, planned):
        from src.modules.m5_physics_engine import Scene, Simulator, CameraConfig, CinematicCamera
        from src.shared.vocabulary import OBJECTS

        scene = Scene(gravity=-9.81)
        scene.setup()
        scene.add_ground()

        for ent in planned.entities:
            obj_def = OBJECTS.get(ent.object_type)
            if obj_def and obj_def.category.name != "HUMANOID":
                scene.add_primitive(
                    name=ent.name,
                    shape=obj_def.default_shape,
                    size=ent.size or obj_def.default_size,
                    mass=ent.mass,
                    position=ent.position.to_list(),
                    color=list(ent.color) if ent.color else [0.5, 0.5, 0.5, 1.0],
                )

        sim = Simulator(scene, CameraConfig(width=640, height=480, distance=2.5))
        cam = CinematicCamera(distance=2.5, yaw=30, pitch=-20)
        cam.add_orbit(start_yaw=0, end_yaw=90, duration=self.config.duration)
        frames = sim.run_cinematic(
            duration=self.config.duration, fps=self.config.fps, cinematic_camera=cam
        )
        return frames, sim, scene

    def _export_video(self, sim, frames: List, output_name: str) -> str:
        videos_dir = os.path.join(self.config.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        path = os.path.join(videos_dir, f"{output_name}.mp4")
        sim.create_video(frames, path, fps=self.config.fps)
        log.info("[M7] video → %s", path)
        return path


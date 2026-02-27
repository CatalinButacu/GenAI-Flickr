from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from src.shared.mem_profile import profile_memory, tracemalloc_snapshot
from src.modules.m7_render_engine import RenderEngine, RenderSettings

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    assets_dir: str = "assets/objects"
    use_t5_parser: bool = True           # M1: prefer T5 model over rules parser
    use_asset_generation: bool = False  # M3: Shap-E (GPU)
    use_motion_generation: bool = True   # M4: SSM / KIT-ML
    use_rl_controller: bool = False      # M6: PPO stub
    use_ai_enhancement: bool = False     # M8: ControlNet (GPU)
    use_render_engine: bool = True        # M7: motion blur, DoF, color grade (CPU)
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

        # Prefer T5SceneParser (ML) over PromptParser (rules) when configured
        if self.config.use_t5_parser:
            try:
                from src.modules.m1_scene_understanding.t5_parser import T5SceneParser
                self._parser = T5SceneParser(device=self.config.device, fallback=True)
                log.info("[M1] T5SceneParser ready (ML-powered, with rules fallback)")
            except Exception as exc:
                log.warning("[M1] T5SceneParser unavailable (%s) — using rules parser", exc)
                self._parser = PromptParser()
                log.info("[M1] PromptParser ready (rules-based)")
        else:
            self._parser = PromptParser()
            log.info("[M1] PromptParser ready (rules-based)")

        self._planner = ScenePlanner()
        log.info("[M2] ScenePlanner ready")

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
        self._motion_gen = MotionGenerator(use_retrieval=True, use_ssm=True)

    def _init_enhancer(self) -> None:
        # Prefer AnimateDiff (temporal consistency) over per-frame ControlNet.
        # Use 384×384 and batch_size=4 to fit within 4 GB VRAM.
        from src.modules.m8_ai_enhancer import AnimateDiffHumanRenderer
        try:
            self._enhancer = AnimateDiffHumanRenderer(
                device=self.config.device,
                batch_size=4,
                width=384,
                height=384,
            )
            if self._enhancer.setup():
                log.info("[M8] AnimateDiff + ControlNet renderer loaded (384×384, batch=4)")
                return
            log.warning("[M8] AnimateDiff setup failed — falling back to per-frame ControlNet")
        except Exception as exc:
            log.warning("[M8] AnimateDiff unavailable (%s) — falling back", exc)

        from src.modules.m8_ai_enhancer import ControlNetHumanRenderer
        self._enhancer = ControlNetHumanRenderer(device=self.config.device)
        if not self._enhancer.setup():
            raise RuntimeError("ControlNet setup failed")

    def run(self, prompt: str, output_name: str = "output") -> Dict[str, Any]:
        if not self._is_setup:
            self.setup()

        with tracemalloc_snapshot("M1 parse"):
            parsed = self._parser.parse(prompt)
        log.info("[M1] %d entities, %d actions", len(parsed.entities), len(parsed.actions))

        with tracemalloc_snapshot("M2 plan"):
            planned = self._planner.plan(parsed)
        log.info("[M2] %d entities positioned", len(planned.entities))

        with tracemalloc_snapshot("M4 motion"):
            motion_clips = self._generate_motion(parsed)
        with tracemalloc_snapshot("M5 physics"):
            frames, sim, scene = self._run_physics(planned, motion_clips)
        log.info("[M5] %d frames captured", len(frames))

        with tracemalloc_snapshot("M7 export"):
            video_path = self._export_video(sim, frames, output_name)
        scene.cleanup()

        if self.config.use_ai_enhancement and self._enhancer:
            log.info("[M8] ControlNet rendering active (integrated in M5 loop)")

        return {
            "prompt": prompt,
            "parsed_scene": parsed,
            "planned_scene": planned,
            "motion_clips": motion_clips,
            "physics_frames": frames,
            "video_path": video_path,
        }

    @profile_memory
    def _generate_motion(self, parsed) -> Dict[str, Any]:
        if not (self.config.use_motion_generation and self._motion_gen):
            return {}
        from src.shared.vocabulary import ACTIONS
        num_frames = int(self.config.duration * 20)  # KIT-ML is at 20 fps
        return {
            action.actor: self._motion_gen.generate(
                ACTIONS[action.action_type].motion_clip.replace("_", " "),
                num_frames=num_frames,
            )
            for action in parsed.actions
            if (act_def := ACTIONS.get(action.action_type)) and act_def.motion_clip
        }

    @profile_memory
    def _run_physics(self, planned, motion_clips: Dict[str, Any] = None):
        from src.modules.m5_physics_engine import (
            Scene, Simulator, CameraConfig, CinematicCamera,
            load_humanoid, retarget_sequence,
        )
        from src.shared.vocabulary import OBJECTS

        scene = Scene(gravity=-9.81)
        scene.setup()
        scene.add_ground()
        self._populate_scene_primitives(scene, planned.entities, OBJECTS)

        has_humanoid = any(
            OBJECTS.get(e.object_type) and
            OBJECTS[e.object_type].category.name == "HUMANOID"
            for e in planned.entities
        )

        humanoid, joint_angles_seq, root_transforms_seq = self._setup_humanoid(
            scene, has_humanoid, motion_clips, load_humanoid, retarget_sequence,
        )

        cam_target = [0.0, 0.3, 1.0] if has_humanoid else [0.0, 0.0, 0.3]
        cam_dist   = 3.5             if has_humanoid else 2.5
        sim = Simulator(scene, CameraConfig(width=640, height=480,
                                            distance=cam_dist, target=cam_target))
        cam = CinematicCamera(target=cam_target, distance=cam_dist, yaw=30, pitch=-20)
        cam.add_orbit(start_yaw=10, end_yaw=70, duration=self.config.duration)

        if has_humanoid and joint_angles_seq:
            frames = self._run_motion_driven(
                sim, humanoid, joint_angles_seq, root_transforms_seq,
                cam, scene.client,
            )
        else:
            frames = sim.run_cinematic(
                duration=self.config.duration, fps=self.config.fps,
                cinematic_camera=cam,
            )
        return frames, sim, scene

    @staticmethod
    def _populate_scene_primitives(scene, entities, objects_vocab) -> None:
        for ent in entities:
            obj_def = objects_vocab.get(ent.object_type)
            if obj_def and obj_def.category.name != "HUMANOID":
                scene.add_primitive(
                    name=ent.name,
                    shape=obj_def.default_shape,
                    size=ent.size or obj_def.default_size,
                    mass=ent.mass,
                    position=ent.position.to_list(),
                    color=list(ent.color) if ent.color else [0.5, 0.5, 0.5, 1.0],
                )

    def _setup_humanoid(self, scene, has_humanoid, motion_clips,
                        load_humanoid_fn, retarget_sequence_fn):
        if not has_humanoid:
            return None, [], []
        humanoid = load_humanoid_fn(scene, position=[0.0, 0.0, 1.0])
        raw_joints = self._pick_raw_joints(motion_clips)
        if raw_joints is None:
            log.warning("[M5] no raw joints found — humanoid held in T-pose")
            return humanoid, [], []
        joint_angles_seq, root_transforms_seq = retarget_sequence_fn(raw_joints)
        log.info("[M5] retargeted %d frames to PyBullet joint angles",
                 len(joint_angles_seq))
        return humanoid, joint_angles_seq, root_transforms_seq

    @staticmethod
    def _pick_raw_joints(motion_clips) -> "Any | None":
        for clip in (motion_clips or {}).values():
            if clip is not None and getattr(clip, "raw_joints", None) is not None:
                log.info("[M5] using motion clip '%s' (%d raw frames)",
                         clip.action, len(clip.raw_joints))
                return clip.raw_joints
        return None

    def _run_motion_driven(
        self,
        sim: Any,
        humanoid: Any,
        joint_angles_seq: list,
        root_transforms_seq: list,
        cam: Any,
        phys_client: int,
    ) -> List:
        """
        Physics-driven simulation loop with two rendering paths:

        **Path A — ControlNet (M8 enabled)**
          1. Run full physics loop, collecting physics-verified skeleton poses.
          2. Project each skeleton → 2-D OpenPose image.
          3. Feed into SD 1.5 + ControlNet OpenPose for photorealistic frames.
          4. M7 post-processes the final RGB frames.

        **Path B — Skeleton renderer (M8 disabled)**
          Same physics loop, but rendered as a cinematic glow skeleton via
          PhysicsSkeletonRenderer (OpenCV-based).

        In *both* paths the 3-D joint positions come from PyBullet *after*
        the physics step — they are the truth output of the rigid-body solver,
        not the raw KIT-ML data.
        """
        import pybullet as p
        from src.modules.m5_physics_engine import (
            PhysicsSkeletonRenderer, physics_links_to_skeleton,
        )

        phys_hz         = sim.physics_hz
        render_interval = phys_hz // self.config.fps
        total_steps     = int(self.config.duration * phys_hz)
        total_frames    = len(joint_angles_seq)

        # ── Collect physics-verified skeleton positions ──────────────────
        skeleton_positions: List = []
        log.info("[M5] physics loop: %d steps, %d motion frames",
                 total_steps, total_frames)

        for step in range(total_steps):
            t         = step / phys_hz
            frame_idx = min(int((t / self.config.duration) * total_frames),
                            total_frames - 1)

            # RSI root tracking
            pos, quat = root_transforms_seq[frame_idx]
            p.resetBasePositionAndOrientation(
                humanoid.body_id, pos, quat, physicsClientId=phys_client
            )
            humanoid.set_joint_position_targets(
                joint_angles_seq[frame_idx], max_force=200.0
            )
            sim.step(dt=1.0 / phys_hz)

            if step % render_interval == 0:
                link_pos = humanoid.get_link_world_positions()
                xyz_21   = physics_links_to_skeleton(link_pos)
                skeleton_positions.append(xyz_21)

        log.info("[M5] collected %d physics-verified skeleton poses",
                 len(skeleton_positions))

        # ── Render frames ────────────────────────────────────────────────
        use_controlnet = (
            self.config.use_ai_enhancement
            and self._enhancer is not None
            and self._enhancer.is_ready
        )

        if use_controlnet:
            raw_frames = self._render_controlnet(
                skeleton_positions,
            )
        else:
            raw_frames = self._render_skeleton(
                skeleton_positions, cam,
            )

        from src.modules.m5_physics_engine import FrameData
        import numpy as np
        frame_h, frame_w = raw_frames[0].shape[:2]
        dummy_depth = np.zeros((frame_h, frame_w), dtype=np.uint8)
        dummy_seg   = np.zeros((frame_h, frame_w), dtype=np.int32)
        return [
            FrameData(timestamp=i / self.config.fps,
                      rgb=f, depth=dummy_depth, segmentation=dummy_seg)
            for i, f in enumerate(raw_frames)
        ]

    # ------------------------------------------------------------------
    def _render_controlnet(
        self, skeleton_positions: List,
    ) -> List:
        """Render physics-verified skeletons via AI enhancer (M8).

        Two sub-strategies depending on which renderer is loaded:

        **AnimateDiff** (preferred):
          Project all skeletons → OpenPose, then feed entire sequence to
          ``AnimateDiffHumanRenderer.render_sequence()`` which handles
          batching (8-frame windows with overlap) and cross-frame temporal
          attention.  No keyframe interpolation needed.

        **Per-frame ControlNet** (fallback):
          Render every 4th frame through ControlNet, linearly interpolate
          RGB between keyframes.  4× speed boost for ~5% quality cost.
        """
        import numpy as np
        from src.modules.m8_ai_enhancer import SkeletonProjector
        from src.modules.m8_ai_enhancer.animatediff_human import (
            AnimateDiffHumanRenderer,
        )

        projector = SkeletonProjector(
            img_w=512, img_h=512, cam_yaw_deg=15.0,
            joint_radius=5, bone_thickness=3,
        )

        # Subsample if too many frames
        max_frames = 32
        positions = skeleton_positions
        if len(positions) > max_frames:
            indices = np.linspace(0, len(positions) - 1,
                                  max_frames, dtype=int)
            positions = [positions[i] for i in indices]
            log.info("[M8] subsampled %d → %d frames",
                     len(skeleton_positions), len(positions))

        # ── AnimateDiff path (batch with temporal attention) ─────────────
        if isinstance(self._enhancer, AnimateDiffHumanRenderer):
            log.info("[M8] projecting %d skeletons → OpenPose images …",
                     len(positions))
            skeleton_images = [projector.render(xyz) for xyz in positions]

            log.info("[M8] AnimateDiff batch rendering (%d frames) …",
                     len(skeleton_images))
            return self._enhancer.render_sequence(skeleton_images)

        # ── Per-frame ControlNet path (keyframe + interpolation) ─────────
        keyframe_interval = 4
        total = len(positions)

        if total <= keyframe_interval * 2:
            keyframe_indices = list(range(total))
        else:
            keyframe_indices = list(range(0, total, keyframe_interval))
            if keyframe_indices[-1] != total - 1:
                keyframe_indices.append(total - 1)

        keyframe_skeletons = [positions[i] for i in keyframe_indices]

        log.info("[M8] projecting %d keyframe skeletons → OpenPose images …",
                 len(keyframe_skeletons))
        skeleton_images = [projector.render(xyz) for xyz in keyframe_skeletons]

        log.info("[M8] ControlNet inference (%d keyframes, %d steps) …",
                 len(skeleton_images), self._enhancer.num_steps)
        keyframes_rgb = self._enhancer.render_sequence(skeleton_images)

        # Interpolate between keyframes
        if len(keyframe_indices) == total:
            return keyframes_rgb

        log.info("[M8] interpolating %d keyframes → %d final frames …",
                 len(keyframes_rgb), total)
        all_frames: List[np.ndarray] = [np.empty(0)] * total

        for k_idx in range(len(keyframe_indices)):
            all_frames[keyframe_indices[k_idx]] = keyframes_rgb[k_idx]

        for seg in range(len(keyframe_indices) - 1):
            start_idx = keyframe_indices[seg]
            end_idx = keyframe_indices[seg + 1]
            start_frame = keyframes_rgb[seg].astype(np.float32)
            end_frame = keyframes_rgb[seg + 1].astype(np.float32)

            gap = end_idx - start_idx
            for j in range(1, gap):
                alpha = j / gap
                blended = ((1.0 - alpha) * start_frame + alpha * end_frame)
                all_frames[start_idx + j] = blended.astype(np.uint8)

        return all_frames

    # ------------------------------------------------------------------
    def _render_skeleton(
        self, skeleton_positions: List, cam: Any,
    ) -> List:
        """Fallback: render physics-verified skeletons as glow skeleton."""
        from src.modules.m5_physics_engine import PhysicsSkeletonRenderer
        import numpy as np

        renderer = PhysicsSkeletonRenderer(
            img_w=720, img_h=1080, yaw_deg=30, pitch_deg=-20,
            distance=3.5, target=[0.0, 0.8, 0.0],
        )

        raw_frames = []
        prev_xyz: "np.ndarray | None" = None
        for i, xyz_21 in enumerate(skeleton_positions):
            t = i / max(len(skeleton_positions) - 1, 1)
            cam.update(t * self.config.duration)
            cyaw, cpitch, cdist, ctarget = cam.get_camera_params()
            renderer.update_camera(cyaw, cpitch, cdist, ctarget)

            frame_rgb = renderer.render_frame(xyz_21, prev_xyz)
            raw_frames.append(frame_rgb)
            prev_xyz = xyz_21

        log.info("[M5] rendered %d skeleton frames", len(raw_frames))
        return raw_frames

    @profile_memory
    def _export_video(self, sim, frames: List, output_name: str) -> str:
        videos_dir = os.path.join(self.config.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        path = os.path.join(videos_dir, f"{output_name}.mp4")
        if self.config.use_render_engine:
            engine = RenderEngine(RenderSettings(
                motion_blur=True,
                dof=True,
                color_grade=True,
                vignette=True,
                film_grain=False,
                output_layout="rgb",
            ))
            engine.render(frames, path, fps=self.config.fps)
        else:
            sim.create_video(frames, path, fps=self.config.fps, layout="rgb")
        log.info("[M7] video → %s", path)
        return path


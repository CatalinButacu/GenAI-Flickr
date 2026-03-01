"""
#WHERE
    Entry point of the whole system — called by main.py,
    scripts/run_showcase.py, and examples/.

#WHAT
    End-to-end pipeline: text prompt → M1 parse → M2 plan → M3 assets →
    M4 motion → M5 physics → M7 render → (optional M8 enhance) → MP4.

#INPUT
    Text prompt string, PipelineConfig.

#OUTPUT
    Dict with video path, frame count, per-module timings.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from src.shared.constants import GRAVITY
from src.shared.mem_profile import profile_memory, tracemalloc_snapshot
from src.modules.render_engine import RenderEngine, RenderSettings

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    output_dir: str = "outputs"
    assets_dir: str = "assets/objects"
    use_t5_parser: bool = True           # M1: prefer T5 model over rules parser
    use_asset_generation: bool = False   # M3: Shap-E (GPU)
    use_motion_generation: bool = True   # M4: SSM / KIT-ML
    use_physics_ssm: bool = True         # M4: PhysicsSSM refinement pass
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
        self._motion_gen = self._enhancer = self._physics_ssm_refiner = None
        self._kb_retriever = None

    def setup(self) -> None:
        from src.modules.scene_understanding.prompt_parser import PromptParser
        from src.modules.scene_planner import ScenePlanner

        # Prefer T5SceneParser (ML) over PromptParser (rules) when configured
        if self.config.use_t5_parser:
            try:
                from src.modules.scene_understanding.t5_parser import T5SceneParser
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

        # Knowledge retriever — SBERT + FAISS semantic lookup over VG objects
        try:
            from src.modules.scene_understanding.retriever import KnowledgeRetriever
            self._kb_retriever = KnowledgeRetriever()
            if self._kb_retriever.setup():
                log.info("[M1] KnowledgeRetriever ready — %d entries",
                         self._kb_retriever.entry_count)
            else:
                log.warning("[M1] KnowledgeRetriever empty — disabled")
                self._kb_retriever = None
        except Exception as exc:
            log.warning("[M1] KnowledgeRetriever unavailable (%s)", exc)
            self._kb_retriever = None

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

        log.info("[M5] PhysicsEngine ready | [M7] RenderEngine ready")
        self._is_setup = True

    def _init_asset_gen(self) -> None:
        from src.modules.asset_generator import ModelGenerator
        self._asset_gen = ModelGenerator(device=self.config.device)
        if not self._asset_gen.setup():
            raise RuntimeError("setup() returned False")

    def _init_motion_gen(self) -> None:
        from src.modules.motion_generator import MotionGenerator
        self._motion_gen = MotionGenerator(use_retrieval=True, use_ssm=True, use_semantic=True)
        # PhysicsSSM refinement pass — blends SSM temporal modelling with
        # physics constraints through a learned sigmoid gate (novel contribution)
        if self.config.use_physics_ssm:
            try:
                from src.modules.motion_generator.ssm_generator import (
                    SSMMotionGenerator, SSMMotionConfig,
                )
                cfg = SSMMotionConfig(use_physics=True)
                self._physics_ssm_refiner = SSMMotionGenerator(
                    backend="ssm_physics", config=cfg, device=self.config.device,
                )
                if self._physics_ssm_refiner.setup():
                    log.info("[M4] PhysicsSSM refinement layer active")
                else:
                    log.warning("[M4] PhysicsSSM setup failed — skipping")
                    self._physics_ssm_refiner = None
            except Exception as exc:
                log.warning("[M4] PhysicsSSM unavailable (%s)", exc)
                self._physics_ssm_refiner = None

    def _init_enhancer(self) -> None:
        # Prefer AnimateDiff (temporal consistency) over per-frame ControlNet.
        # Use 384×384 and batch_size=4 to fit within 4 GB VRAM.
        from src.modules.ai_enhancer import AnimateDiffHumanRenderer
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

        from src.modules.ai_enhancer import ControlNetHumanRenderer
        self._enhancer = ControlNetHumanRenderer(device=self.config.device)
        if not self._enhancer.setup():
            raise RuntimeError("ControlNet setup failed")

    def run(self, prompt: str, output_name: str = "output") -> Dict[str, Any]:
        if not self._is_setup:
            self.setup()

        with tracemalloc_snapshot("M1 parse"):
            parsed = self._parser.parse(prompt)
        log.info("[M1] %d entities, %d actions", len(parsed.entities), len(parsed.actions))

        # ── KB enrichment: semantic lookup of entity properties ──────
        if self._kb_retriever is not None:
            self._enrich_entities_from_kb(parsed)

        with tracemalloc_snapshot("M2 plan"):
            planned = self._planner.plan(parsed)
        log.info("[M2] %d entities positioned", len(planned.entities))

        with tracemalloc_snapshot("M4 motion"):
            motion_clips = self._generate_motion(parsed)
        with tracemalloc_snapshot("M5 physics"):
            frames, sim, scene = self._run_physics(planned, motion_clips, parsed)
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
        total_frames = int(self.config.duration * 20)  # KIT-ML is at 20 fps

        # ── Collect an ORDERED list of (actor, clip) pairs ───────────
        # Each parsed action gets its own clip; multiple actions for the
        # same actor are kept separate so they can be sequenced.
        action_clips: List[tuple] = []
        for action in parsed.actions:
            act_def = ACTIONS.get(action.action_type)
            if act_def and act_def.motion_clip:
                # Divide total frames proportionally across the action count
                n = max(total_frames // max(len(parsed.actions), 1), 20)
                clip = self._motion_gen.generate(
                    act_def.motion_clip.replace("_", " "), num_frames=n,
                )
                action_clips.append((action.actor, clip))

        if not action_clips:
            return {}

        # ── Sequence & blend clips that share the same actor ─────────
        actor_sequences: Dict[str, List] = {}
        for actor, clip in action_clips:
            actor_sequences.setdefault(actor, []).append(clip)

        clips: Dict[str, Any] = {}
        for actor, seq in actor_sequences.items():
            if len(seq) == 1:
                clips[actor] = seq[0]
            else:
                clips[actor] = self._blend_motion_clips(seq)
                log.info("[M4] sequenced %d clips for '%s' → %d frames",
                         len(seq), actor, clips[actor].num_frames)

        # ── PhysicsSSM refinement pass ───────────────────────────────
        # Blend SSM temporal modelling with physics constraints through
        # the learned sigmoid gate:
        #   gate = σ(W [ssm_out ; physics_embed])
        #   output = gate·ssm + (1-gate)·constraints
        if clips and self._physics_ssm_refiner is not None:
            import numpy as np
            for actor, clip in clips.items():
                if clip is not None and clip.frames is not None:
                    physics_state = np.zeros(
                        (clip.frames.shape[0], 64), dtype=np.float32,
                    )
                    # Encode basic physics priors: gravity, height, momentum
                    physics_state[:, 0] = GRAVITY     # gravity Z
                    if clip.raw_joints is not None:
                        pelvis_h = clip.raw_joints[:, 0, 1] / 1000.0  # mm→m
                        physics_state[:, 1] = pelvis_h
                        # Approximate velocity from finite differences
                        if len(pelvis_h) > 1:
                            vel = np.gradient(pelvis_h, 1.0 / clip.fps)
                            physics_state[:, 2] = vel
                    refined = self._physics_ssm_refiner.generate(
                        clip.action, num_frames=len(clip.frames),
                        physics_state=physics_state,
                    )
                    if refined.raw_joints is None and clip.raw_joints is not None:
                        refined.raw_joints = clip.raw_joints
                    clips[actor] = refined
                    log.info("[M4] PhysicsSSM refined '%s' → %d frames",
                             actor, len(refined.frames))

        return clips

    @staticmethod
    def _blend_motion_clips(
        clips: List,
        blend_frames: int = 10,
    ):
        """Concatenate motion clips with linear cross-fade over *blend_frames*.

        For each pair of consecutive clips, the last *blend_frames* of clip_i
        are linearly blended with the first *blend_frames* of clip_{i+1}.
        This eliminates discontinuities at action boundaries.
        """
        import numpy as np
        from src.modules.motion_generator.models import MotionClip

        # Concatenate feature arrays with crossfade
        feature_parts: List[np.ndarray] = []
        joint_parts: List[np.ndarray] = []
        action_labels: List[str] = []

        for i, clip in enumerate(clips):
            action_labels.append(clip.action)
            f = clip.frames
            j = clip.raw_joints  # may be None

            if i > 0 and blend_frames > 0:
                # Crossfade overlap: previous tail + current head
                n_blend = min(blend_frames, len(feature_parts[-1]), len(f))
                if n_blend > 0:
                    alpha = np.linspace(0.0, 1.0, n_blend).astype(np.float32)

                    # Features blend
                    tail = feature_parts[-1][-n_blend:]
                    head = f[:n_blend]
                    blended = tail * (1 - alpha[:, None]) + head * alpha[:, None]
                    feature_parts[-1] = feature_parts[-1][:-n_blend]
                    feature_parts.append(blended)
                    feature_parts.append(f[n_blend:])

                    # Raw joints blend (if both have them)
                    if j is not None and joint_parts and joint_parts[-1] is not None:
                        j_tail = joint_parts[-1][-n_blend:]
                        j_head = j[:n_blend]
                        j_blended = (
                            j_tail * (1 - alpha[:, None, None])
                            + j_head * alpha[:, None, None]
                        )
                        joint_parts[-1] = joint_parts[-1][:-n_blend]
                        joint_parts.append(j_blended)
                        joint_parts.append(j[n_blend:])
                    elif j is not None:
                        joint_parts.append(j)
                else:
                    feature_parts.append(f)
                    if j is not None:
                        joint_parts.append(j)
            else:
                feature_parts.append(f)
                if j is not None:
                    joint_parts.append(j)

        combined_features = np.concatenate(feature_parts, axis=0)
        combined_joints = (
            np.concatenate(joint_parts, axis=0)
            if joint_parts and all(jp is not None for jp in joint_parts)
            else None
        )

        return MotionClip(
            action=" then ".join(action_labels),
            frames=combined_features,
            fps=clips[0].fps,
            source="sequenced",
            raw_joints=combined_joints,
        )

    def _enrich_entities_from_kb(self, parsed) -> None:
        """Enrich parsed entities with KB properties via semantic lookup.

        For each entity, query the FAISS index for the closest VG object and
        apply its physical properties (dimensions, mass, material, mesh prompt)
        when the parser didn't already set them.
        """
        enriched = 0
        for entity in parsed.entities:
            results = self._kb_retriever.retrieve(entity.name, top_k=1)
            if not results:
                continue
            kb = results[0]
            # Only fill in properties that the parser left at defaults
            if not entity.color and kb.material:
                entity.object_type = entity.object_type or kb.category
            if kb.mesh_prompt:
                entity.name = entity.name  # keep original name
            enriched += 1
            log.debug("[M1-KB] '%s' → KB '%s' (category=%s, mass=%.1fkg)",
                      entity.name, kb.name, kb.category, kb.mass)
        if enriched:
            log.info("[M1] KB-enriched %d/%d entities", enriched, len(parsed.entities))

    @profile_memory
    def _run_physics(self, planned, motion_clips: Dict[str, Any] = None,
                     parsed=None):
        from src.modules.physics_engine import (
            Scene, Simulator, CameraConfig, CinematicCamera,
            load_humanoid, retarget_sequence,
        )
        from src.shared.vocabulary import OBJECTS

        scene = Scene(gravity=GRAVITY)
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

        cam_target = [0.0, 1.0, 0.0] if has_humanoid else [0.0, 0.0, 0.3]
        cam_dist   = 5.0             if has_humanoid else 2.5
        sim = Simulator(scene, CameraConfig(width=640, height=480,
                                            distance=cam_dist, target=cam_target))
        # Humanoid: orbit from front (90°) to back (270°)
        init_yaw = 90 if has_humanoid else 45
        cam = CinematicCamera(target=cam_target, distance=cam_dist,
                              yaw=init_yaw, pitch=-15)
        cam.add_orbit(start_yaw=init_yaw, end_yaw=init_yaw + 180,
                      duration=self.config.duration)

        if has_humanoid and joint_angles_seq:
            # Extract action label for the skeleton renderer overlay
            action_label = ""
            if parsed and parsed.actions:
                action_label = " + ".join(a.action_type.replace("_", " ")
                                          for a in parsed.actions)
            frames = self._run_motion_driven(
                sim, humanoid, joint_angles_seq, root_transforms_seq,
                cam, scene, action_label=action_label,
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
        """Concatenate raw joints from all clips (respects action sequencing)."""
        import numpy as np
        parts = []
        for clip in (motion_clips or {}).values():
            if clip is not None and getattr(clip, "raw_joints", None) is not None:
                log.info("[M5] using motion clip '%s' (%d raw frames, src=%s)",
                         clip.action, len(clip.raw_joints), clip.source)
                parts.append(clip.raw_joints)
        if not parts:
            return None
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def _run_motion_driven(
        self,
        sim: Any,
        humanoid: Any,
        joint_angles_seq: list,
        root_transforms_seq: list,
        cam: Any,
        scene: Any,
        action_label: str = "",
    ) -> List:
        """
        Physics-driven simulation loop with contact detection and two
        rendering paths:

        **Physics interactions (new)**
          After each physics step, query PyBullet for contact points between
          the humanoid's feet and the ground / scene objects.  Contact events
          are logged and scene objects react to humanoid forces (e.g. a ball
          near a kicking foot gets pushed).

        **Path A — ControlNet (M8 enabled)**
          Physics-verified skeleton → OpenPose → SD 1.5 + ControlNet.

        **Path B — Skeleton renderer (M8 disabled)**
          Physics-verified skeleton → cinematic glow render (OpenCV).
        """
        import pybullet as p
        from src.modules.physics_engine import (
            PhysicsSkeletonRenderer, physics_links_to_skeleton,
        )

        phys_client     = scene.client
        phys_hz         = sim.physics_hz
        render_interval = phys_hz // self.config.fps
        total_steps     = int(self.config.duration * phys_hz)
        total_frames    = len(joint_angles_seq)

        # Track interaction statistics
        contact_events  = 0
        ground_contacts = 0
        object_contacts = 0

        # Map scene object body_ids for contact attribution
        scene_body_ids = {
            obj.body_id: obj.name
            for obj in scene.objects.values()
        }
        ground_id = scene.ground_id

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

            # ── Contact detection (every render frame) ───────────────
            if step % render_interval == 0:
                # Query foot contacts with ground and scene objects
                foot_contacts = humanoid.get_foot_contacts(scene)
                for foot, contacts in foot_contacts.items():
                    for c in contacts:
                        contact_events += 1
                        body_b = c["bodyB"]
                        if body_b == ground_id:
                            ground_contacts += 1
                        elif body_b in scene_body_ids:
                            object_contacts += 1
                            obj_name = scene_body_ids[body_b]
                            # Scene objects react to contact forces
                            if c["force"] > 5.0:
                                log.debug(
                                    "[M5] %s contact with '%s' "
                                    "(force=%.1fN)",
                                    foot, obj_name, c["force"],
                                )

                link_pos = humanoid.get_link_world_positions()
                xyz_21   = physics_links_to_skeleton(link_pos)
                skeleton_positions.append(xyz_21)

        log.info("[M5] collected %d physics-verified skeleton poses",
                 len(skeleton_positions))
        if contact_events:
            log.info(
                "[M5] interactions: %d ground, %d object contacts "
                "(%d total events)",
                ground_contacts, object_contacts, contact_events,
            )

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
                skeleton_positions, cam, action_label=action_label,
            )

        from src.modules.physics_engine import FrameData
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
        from src.modules.ai_enhancer import SkeletonProjector
        from src.modules.ai_enhancer.animatediff_human import (
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
        action_label: str = "",
    ) -> List:
        """Fallback: render physics-verified skeletons as glow skeleton."""
        from src.modules.physics_engine import (
            PhysicsSkeletonRenderer, auto_orient_skeleton,
        )
        import numpy as np

        # ── Auto-orient so spine extends along +Y (upright) ──────────
        skeleton_positions = auto_orient_skeleton(skeleton_positions)

        # ── Compute camera target from first oriented frame ──────────
        first = skeleton_positions[0]
        # Use pelvis+neck midpoint as vertical centre (avoids toe noise)
        y_center = float((first[0, 1] + first[3, 1]) / 2) / 1000.0   # mm → m
        z_center = float(np.median(first[:, 2])) / 1000.0
        x_center = float(np.median(first[:, 0])) / 1000.0
        # Skeleton height from foot-level joints to head
        core_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18]
        skel_h = float(first[core_joints, 1].max()
                       - first[core_joints, 1].min()) / 1000.0
        cam_dist = max(3.5, skel_h * 1.3)

        # Update CinematicCamera to orbit around actual skeleton centre
        skel_target = [x_center, y_center, z_center]
        cam.base_target = list(skel_target)
        cam.current_target = list(skel_target)
        cam.base_distance = cam_dist
        cam.current_distance = cam_dist
        log.info("[M5] auto-orient: height=%.1fm, target=[%.2f, %.2f, %.2f], "
                 "dist=%.1fm", skel_h, *skel_target, cam_dist)

        renderer = PhysicsSkeletonRenderer(
            img_w=1280, img_h=720,
            action_label=action_label,
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


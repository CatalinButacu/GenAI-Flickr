"""M5 (Physics Simulation) stage — PyBullet sim with humanoid control."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pybullet as p

from src.shared.constants import GRAVITY
from src.shared.mem_profile import profile_memory
from src.shared.vocabulary import OBJECTS
from src.modules.physics import (
    Scene, Simulator, CameraConfig, CinematicCamera, FrameData,
    load_humanoid, retarget_sequence, physics_links_to_skeleton,
)
from src.modules.physics.body_model import BodyParams, body_params_from_prompt

log = logging.getLogger(__name__)


@dataclass
class PhysicsResult:
    """Output of the physics stage."""
    sim: Simulator
    scene: Scene
    cam: CinematicCamera
    action_label: str = ""
    body_params: BodyParams | None = None
    # Humanoid path: raw skeleton positions needing mesh rendering
    skeleton_positions: list | None = None
    # Raw SMPL-X motion data for high-quality SMPLSequence rendering
    smplx_motion: np.ndarray | None = None    # (T, 168)
    smplx_betas: np.ndarray | None = None     # (16,)
    # Non-humanoid path: pre-rendered FrameData
    frames: list[FrameData] | None = None

    @property
    def needs_rendering(self) -> bool:
        return self.skeleton_positions is not None


class PhysicsStage:
    """M5: Run physics-driven simulation and collect skeleton poses."""

    def __init__(
        self, *,
        fps: int = 24,
        duration: float = 5.0,
        fixed_camera: bool = False,
    ) -> None:
        self.fps = fps
        self.duration = duration
        self.fixed_camera = fixed_camera

    # ── public API ──────────────────────────────────────────────────

    @profile_memory
    def run(
        self,
        planned,
        motion_clips: dict[str, Any] | None = None,
        parsed=None,
    ) -> PhysicsResult:
        """Set up scene, run physics, return PhysicsResult."""
        scene = Scene(gravity=GRAVITY)
        scene.setup()
        scene.add_ground()
        self._add_scene_objects(scene, planned.entities)

        has_humanoid = self._has_humanoid(planned.entities)
        humanoid, angles_seq, root_seq = self._setup_humanoid(
            scene, has_humanoid, motion_clips,
        )

        sim, cam = self._build_sim_and_camera(scene, has_humanoid)

        if has_humanoid and angles_seq:
            skeleton_poses, contacts = self._sim_loop(
                sim, humanoid, angles_seq, root_seq, scene,
            )
            self._log_contacts(skeleton_poses, contacts)
            smplx_motion, smplx_betas = _pick_smplx_motion(motion_clips)
            return PhysicsResult(
                sim=sim, scene=scene, cam=cam,
                action_label=self._action_label(parsed),
                body_params=self._extract_body_params(planned, parsed),
                skeleton_positions=skeleton_poses,
                smplx_motion=smplx_motion,
                smplx_betas=smplx_betas,
            )
        else:
            frames = sim.run_cinematic(
                duration=self.duration, fps=self.fps,
                cinematic_camera=cam,
            )
            return PhysicsResult(
                sim=sim, scene=scene, cam=cam, frames=frames,
            )

    # ── scene setup ─────────────────────────────────────────────────

    @staticmethod
    def _add_scene_objects(scene: Scene, entities) -> None:
        for ent in entities:
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

    @staticmethod
    def _has_humanoid(entities) -> bool:
        return any(
            OBJECTS.get(e.object_type)
            and OBJECTS[e.object_type].category.name == "HUMANOID"
            for e in entities
        )

    def _setup_humanoid(self, scene, has_humanoid, motion_clips):
        if not has_humanoid:
            return None, [], []
        humanoid = load_humanoid(scene, position=[0.0, 0.0, 0.9])
        raw_joints = _pick_raw_joints(motion_clips)
        if raw_joints is None:
            log.warning("[M5] no raw joints — humanoid held in T-pose")
            return humanoid, [], []
        angles, roots = retarget_sequence(raw_joints)
        log.info("[M5] retargeted %d frames to PyBullet", len(angles))
        return humanoid, angles, roots

    def _build_sim_and_camera(self, scene, has_humanoid):
        target = [0.0, 1.0, 0.0] if has_humanoid else [0.0, 0.0, 0.3]
        dist = 5.0 if has_humanoid else 2.5
        sim = Simulator(scene, CameraConfig(
            width=640, height=480, distance=dist, target=target,
        ))
        yaw = 90 if has_humanoid else 45
        cam = CinematicCamera(target=target, distance=dist, yaw=yaw, pitch=-15)
        if not self.fixed_camera:
            cam.add_orbit(start_yaw=yaw, end_yaw=yaw + 45,
                          duration=self.duration)
            cam.add_pitch(start_pitch=-15, end_pitch=-20,
                          duration=self.duration)
        return sim, cam

    @staticmethod
    def _extract_body_params(planned, parsed) -> BodyParams | None:
        for e in planned.entities:
            if e.is_actor and getattr(e, 'body_params', None) is not None:
                return e.body_params
        if parsed:
            return body_params_from_prompt(parsed.prompt)
        return None

    @staticmethod
    def _action_label(parsed) -> str:
        if parsed and parsed.actions:
            return " + ".join(
                a.action_type.replace("_", " ") for a in parsed.actions
            )
        return ""

    # ── simulation loop ─────────────────────────────────────────────

    def _sim_loop(self, sim, humanoid, angles_seq, root_seq, scene):
        """Run physics steps and gather skeleton positions + contacts."""
        phys_hz = sim.physics_hz
        render_interval = phys_hz // self.fps
        total_steps = int(self.duration * phys_hz)
        total_frames = len(angles_seq)

        skeleton_positions: list = []
        ground_contacts = object_contacts = contact_events = 0

        body_ids = {o.body_id: o.name for o in scene.objects.values()}
        ground_id = scene.ground_id

        # Hot-path lookups
        _reset = p.resetBasePositionAndOrientation
        _set = humanoid.set_joint_position_targets
        _step = sim.step
        _links = humanoid.get_link_world_positions
        _ratio = total_frames / total_steps
        _hid = humanoid.body_id
        _client = scene.client

        for step in range(total_steps):
            idx = min(int(step * _ratio), total_frames - 1)
            pos, quat = root_seq[idx]
            _reset(_hid, pos, quat, physicsClientId=_client)
            _set(angles_seq[idx])
            _step(dt=1.0 / phys_hz)

            if step % render_interval == 0:
                gc, oc, ce = _process_contacts(
                    humanoid, scene, ground_id, body_ids,
                )
                ground_contacts += gc
                object_contacts += oc
                contact_events += ce
                skeleton_positions.append(physics_links_to_skeleton(_links()))

        return skeleton_positions, (contact_events, ground_contacts, object_contacts)

    @staticmethod
    def _log_contacts(skeleton_positions, contact_stats) -> None:
        ce, gc, oc = contact_stats
        log.info("[M5] collected %d skeleton poses", len(skeleton_positions))
        if ce:
            log.info("[M5] contacts: %d ground, %d object (%d total)",
                     gc, oc, ce)


# ── module-level helpers ───────────────────────────────────────────────

def _pick_raw_joints(motion_clips) -> Any:
    parts = []
    for clip in (motion_clips or {}).values():
        if clip is not None and getattr(clip, "raw_joints", None) is not None:
            parts.append(clip.raw_joints)
    if not parts:
        return None
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _pick_smplx_motion(motion_clips) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract raw (T, 168) SMPL-X params and betas from motion clips."""
    parts = []
    betas = None
    for clip in (motion_clips or {}).values():
        if clip is not None and getattr(clip, "frames", None) is not None:
            parts.append(clip.frames)
            if betas is None and getattr(clip, "betas", None) is not None:
                betas = clip.betas
    if not parts:
        return None, None
    motion = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return motion, betas


def _process_contacts(humanoid, scene, ground_id, body_ids):
    gc = oc = ce = 0
    for _foot, contacts in humanoid.get_foot_contacts(scene).items():
        for c in contacts:
            ce += 1
            body_b = c["bodyB"]
            if body_b == ground_id:
                gc += 1
            elif body_b in body_ids:
                oc += 1
    return gc, oc, ce

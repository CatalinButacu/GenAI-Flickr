"""Humanoid body loader and joint controller for PyBullet (M5 extension)."""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data
import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class HumanoidConfig:
    height: float             = 1.7
    mass: float               = 70.0
    use_self_collision: bool  = False
    fixed_base: bool          = False


@dataclass(slots=True)
class HumanoidState:
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]


class HumanoidBody:
    """Loads and drives a humanoid URDF inside a PyBullet scene."""

    JOINT_INDICES = {
        "abdomen_x": 0, "abdomen_y": 1, "abdomen_z": 2,
        "right_hip_x": 3, "right_hip_y": 4, "right_hip_z": 5,
        "right_knee": 6, "right_ankle": 7,
        "left_hip_x": 8,  "left_hip_y": 9, "left_hip_z": 10,
        "left_knee": 11,  "left_ankle": 12,
        "right_shoulder_x": 13, "right_shoulder_y": 14, "right_elbow": 15,
        "left_shoulder_x": 16,  "left_shoulder_y": 17,  "left_elbow": 18,
    }

    def __init__(self, config: HumanoidConfig = None):
        self.config          = config or HumanoidConfig()
        self.body_id: Optional[int]       = None
        self._client: Optional[int]       = None
        self._joint_info: Dict[str, dict] = {}

    def load(self, physics_client: int,
             position: List[float] = None,
             orientation: List[float] = None) -> int:
        self._client = physics_client
        position    = position    or [0, 0, 1.0]
        orientation = orientation or [0, 0, 0, 1]
        base = pybullet_data.getDataPath()
        urdf = os.path.join(base, "humanoid", "humanoid.urdf")
        if not os.path.exists(urdf):
            urdf = os.path.join(base, "humanoid_symmetric.urdf")
        self.body_id = p.loadURDF(
            urdf,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=self.config.fixed_base,
            flags=p.URDF_USE_SELF_COLLISION if self.config.use_self_collision else 0,
            physicsClientId=physics_client,
        )
        self._joint_info = {}
        for i in range(p.getNumJoints(self.body_id, physicsClientId=physics_client)):
            info = p.getJointInfo(self.body_id, i, physicsClientId=physics_client)
            self._joint_info[info[1].decode()] = {
                "index": i, "type": info[2],
                "lower": info[8], "upper": info[9],
                "max_force": info[10], "max_vel": info[11],
            }
        log.info("HumanoidBody: %d joints", len(self._joint_info))
        return self.body_id

    def set_joint_positions(self, positions: Dict[str, float]) -> None:
        """Teleport joints (kinematic, no physics)."""
        for name, angle in positions.items():
            if name in self._joint_info:
                p.resetJointState(self.body_id, self._joint_info[name]["index"],
                                  angle, physicsClientId=self._client)

    def set_joint_position_targets(self, positions: Dict[str, float], max_force: float = 100.0) -> None:
        """Drive joints with PD motor control (physics-aware)."""
        for name, target in positions.items():
            if name in self._joint_info:
                p.setJointMotorControl2(
                    self.body_id, self._joint_info[name]["index"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target, force=max_force,
                    physicsClientId=self._client,
                )

    def get_state(self) -> HumanoidState:
        pos, orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._client)
        jp, jv = {}, {}
        for name, info in self._joint_info.items():
            s = p.getJointState(self.body_id, info["index"], physicsClientId=self._client)
            jp[name], jv[name] = s[0], s[1]
        return HumanoidState(position=pos, orientation=orn, joint_positions=jp, joint_velocities=jv)

    def get_link_world_positions(self) -> Dict[str, np.ndarray]:
        """
        Return the world-space 3D position (CoM) of every link after physics.

        Key "base" = root pelvis.
        Remaining keys are the joint names whose child link position is returned.
        All positions in metres, Z-up (PyBullet world frame).
        """
        positions: Dict[str, np.ndarray] = {}
        base_pos, _ = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self._client
        )
        positions["base"] = np.array(base_pos, dtype=np.float64)
        for name, info in self._joint_info.items():
            link_state = p.getLinkState(
                self.body_id, info["index"], physicsClientId=self._client
            )
            # linkWorldPosition is index 0 (CoM frame)
            positions[name] = np.array(link_state[0], dtype=np.float64)
        return positions

    def apply_motion_frame(self, joint_angles: Dict[str, float], use_motors: bool = True) -> None:
        (self.set_joint_position_targets if use_motors else self.set_joint_positions)(joint_angles)

    def get_joint_names(self) -> List[str]:
        return list(self._joint_info.keys())

    def reset_pose(self, pose: str = "t_pose") -> None:
        poses = {
            "t_pose": dict.fromkeys(self._joint_info, 0.0),
            "stand": {"left_hip_x": 0.0, "left_hip_y": 0.0, "left_knee": 0.0,
                      "right_hip_x": 0.0, "right_hip_y": 0.0, "right_knee": 0.0},
        }
        if pose in poses:
            self.set_joint_positions(poses[pose])
        else:
            log.warning("Unknown pose: %s", pose)


def load_humanoid(scene, name: str = "humanoid_1", position: List[float] = None) -> HumanoidBody:
    if not scene._is_setup:
        scene.setup()
    humanoid = HumanoidBody()
    body_id  = humanoid.load(physics_client=scene.client, position=position or [0, 0, 1.0])
    log.info("Loaded humanoid '%s' (id=%d)", name, body_id)
    return humanoid

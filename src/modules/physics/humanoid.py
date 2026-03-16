
from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import math

import pybullet as p
import pybullet_data
import numpy as np

log = logging.getLogger(__name__)

# The default PyBullet humanoid URDF is Y-up and ~5.96 m tall.
# URDF_SCALE  brings it to real human height (~1.73 m).
# Q_UPRIGHT   rotates the Y-up URDF to stand in the Z-up world.
URDF_SCALE = 0.29
Q_UPRIGHT = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))


@dataclass(slots=True)
class HumanoidConfig:
    height: float             = 1.7
    mass: float               = 70.0
    use_self_collision: bool  = True
    fixed_base: bool          = False


class HumanoidBody:
    """Loads and drives a humanoid URDF inside a PyBullet scene.

    The default PyBullet humanoid URDF uses:
      - Spherical (type=2) joints for: chest, neck, shoulders, hips, ankles
      - Revolute  (type=0) joints for: elbows, knees
      - Fixed     (type=4) joints for: root, wrists

    Spherical joints require quaternion targets via setJointMotorControlMultiDof.
    Revolute joints use scalar targets via setJointMotorControl2.
    """

    def __init__(self, config: HumanoidConfig = None):
        self.config          = config or HumanoidConfig()
        self.body_id: int | None       = None
        self._client: int | None       = None
        self._joint_info: dict[str, dict] = {}

    def load(self, physics_client: int,
             position: list[float] = None,
             orientation: list[float] = None) -> int:
        self._client = physics_client
        position    = position    or [0, 0, 0.9]
        orientation = orientation or list(Q_UPRIGHT)
        base = pybullet_data.getDataPath()
        urdf = os.path.join(base, "humanoid", "humanoid.urdf")
        if not os.path.exists(urdf):
            urdf = os.path.join(base, "humanoid_symmetric.urdf")
        self.body_id = p.loadURDF(
            urdf,
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=self.config.fixed_base,
            globalScaling=URDF_SCALE,
            flags=(p.URDF_USE_SELF_COLLISION
                   | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                  if self.config.use_self_collision else 0,
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

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """Teleport joints (kinematic, no physics)."""
        for name, angle in positions.items():
            if name in self._joint_info:
                p.resetJointState(self.body_id, self._joint_info[name]["index"],
                                  angle, physicsClientId=self._client)

    def set_joint_position_targets(self, positions: dict, **_kw) -> None:
        """Set joint targets kinematically (instant, exact positioning).

        *positions* maps URDF joint names to targets:
          - Spherical joints (type=2): target is a 4-tuple quaternion (x,y,z,w)
          - Revolute  joints (type=0): target is a float angle in radians
        Entries for fixed joints (type=4) are silently ignored.
        """
        for name, target in positions.items():
            if name not in self._joint_info:
                continue
            info = self._joint_info[name]
            jtype = info["type"]
            idx = info["index"]
            if jtype == 2:  # spherical → quaternion
                quat = target if len(target) == 4 else (0, 0, 0, 1)
                p.resetJointStateMultiDof(
                    self.body_id, idx,
                    targetValue=list(quat),
                    physicsClientId=self._client,
                )
            elif jtype == 0:  # revolute → scalar
                p.resetJointState(
                    self.body_id, idx,
                    targetValue=float(target),
                    physicsClientId=self._client,
                )

    def get_link_world_positions(self) -> dict[str, np.ndarray]:
        """
        Return the world-space 3D position (CoM) of every link after physics.

        Key "base" = root pelvis.
        Remaining keys are the joint names whose child link position is returned.
        All positions in metres, Z-up (PyBullet world frame).
        """
        positions: dict[str, np.ndarray] = {}
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

    def get_foot_contacts(self, _scene=None) -> dict[str, list[dict]]:
        """Query ground and object contacts for feet (ankles).

        Returns dict with 'left_ankle' and 'right_ankle' keys,
        each containing a list of contact dicts.
        """
        contacts = {}
        for foot in ("left_ankle", "right_ankle"):
            if foot in self._joint_info:
                link_idx = self._joint_info[foot]["index"]
                raw = p.getContactPoints(
                    self.body_id, linkIndexA=link_idx,
                    physicsClientId=self._client,
                )
                contacts[foot] = [
                    {
                        "bodyB": c[2], "posA": c[5], "posB": c[6],
                        "normal": c[7], "distance": c[8], "force": c[9],
                    }
                    for c in raw
                ]
        return contacts


def load_humanoid(scene, name: str = "humanoid_1", position: list[float] = None) -> HumanoidBody:
    if not scene._is_setup:
        scene.setup()
    humanoid = HumanoidBody()
    body_id  = humanoid.load(physics_client=scene.client, position=position or [0, 0, 0.9])
    log.info("Loaded humanoid '%s' (id=%d)", name, body_id)
    return humanoid

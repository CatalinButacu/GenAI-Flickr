from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

import importlib.util

import pybullet as p
import pybullet_data

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PhysicsObject:
    name: str
    body_id: int = -1
    mass: float = 1.0
    position: list[float] = field(default_factory=lambda: [0, 0, 0])
    orientation: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    color: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    is_static: bool = False


def create_shape(shape: str, size: list[float], color: list[float]) -> tuple[int, int]:
    """Create a PyBullet collision+visual shape pair via match/case dispatch."""
    match shape:
        case "box":
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        case "sphere":
            radius = size[0] if isinstance(size, list) else size
            collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        case "cylinder":
            radius = size[0]
            height = size[1] if len(size) > 1 else size[0] * 2
            collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
            visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        case _:
            raise ValueError(f"Unknown shape: {shape}")
    return collision, visual


class Scene:
    def __init__(self, gravity: float = -9.81):  # default matches shared.constants.GRAVITY
        self.gravity = gravity
        self.client = None
        self.objects: dict[str, PhysicsObject] = {}
        self.ground_id = None
        self._is_setup = False
    
    def setup(self, use_gui: bool = False) -> bool:

        try:
            mode = p.GUI if use_gui else p.DIRECT
            self.client = p.connect(mode)
            
            if self.client < 0:
                log.error("PyBullet connection failed")
                return False

            # Try GPU-accelerated offscreen rendering via EGL plugin
            if not use_gui:
                try:
                    if (egl_spec := importlib.util.find_spec("eglRenderer")) and egl_spec.origin:
                        p.loadPlugin(egl_spec.origin, "_eglRendererPlugin")
                        log.info("EGL GPU renderer loaded — fast offscreen rendering")
                except Exception:
                    log.debug("EGL renderer not available — using CPU TinyRenderer")

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, self.gravity)
            p.setRealTimeSimulation(0)
            
            self._is_setup = True
            log.info("Physics scene ready (gravity=%s)", self.gravity)
            return True
        except Exception as e:
            log.error("Scene setup failed: %s", e)
            return False
    
    def add_ground(self) -> int:
        self.ground_id = p.loadURDF("plane.urdf")
        return self.ground_id
    
    def add_primitive(self, name: str, shape: str = "box", size: list[float] = None,
                     mass: float = 1.0, position: list[float] = None,
                     color: list[float] = None, is_static: bool = False) -> PhysicsObject:
        
        size = size or [0.1, 0.1, 0.1]
        position = position or [0, 0, 0.5]
        color = color or [0.5, 0.5, 0.5, 1.0]
        
        collision, visual = create_shape(shape, size, color)
        
        actual_mass = 0 if is_static else mass
        body_id = p.createMultiBody(
            baseMass=actual_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )
        
        obj = PhysicsObject(name=name, body_id=body_id, mass=mass,
                           position=position, color=color, is_static=is_static)
        self.objects[name] = obj
        log.info("Added %s '%s'", shape, name)
        return obj
    
    def load_mesh(self, name: str, mesh_path: str, mass: float = 1.0,
                  position: list[float] = None, scale: float = 1.0,
                  is_static: bool = False) -> PhysicsObject | None:
        
        if not os.path.exists(mesh_path):
            log.error("Mesh not found: %s", mesh_path)
            return None
        
        position = position or [0, 0, 0.5]
        
        collision = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale] * 3)
        visual = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=[scale] * 3)
        
        actual_mass = 0 if is_static else mass
        body_id = p.createMultiBody(
            baseMass=actual_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position
        )
        
        obj = PhysicsObject(name=name, body_id=body_id, mass=mass, position=position, is_static=is_static)
        self.objects[name] = obj
        log.info("Loaded mesh '%s'", name)
        return obj
    
    def get_object(self, name: str) -> PhysicsObject | None:
        return self.objects.get(name)
    
    def get_object_state(self, name: str) -> tuple[list[float], list[float]]:
        if (obj := self.objects.get(name)) is None:
            return [0, 0, 0], [0, 0, 0, 1]
        pos, orn = p.getBasePositionAndOrientation(obj.body_id)
        return list(pos), list(orn)
    
    def apply_force(self, name: str, force: list[float], position: list[float] = None) -> bool:
        if (obj := self.objects.get(name)) is None:
            return False
        position = position or [0, 0, 0]
        p.applyExternalForce(obj.body_id, -1, forceObj=force, posObj=position, flags=p.WORLD_FRAME)
        return True
    
    def set_velocity(self, name: str, linear: list[float], angular: list[float] = None):
        if (obj := self.objects.get(name)) is None:
            return
        angular = angular or [0, 0, 0]
        p.resetBaseVelocity(obj.body_id, linearVelocity=linear, angularVelocity=angular)
    
    def cleanup(self):
        if self.client is not None:
            p.disconnect()
            self.client = None
            self._is_setup = False
            log.info("Scene cleaned up")

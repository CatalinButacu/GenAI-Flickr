"""Physics scene management with Factory pattern for shape creation."""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PhysicsObject:
    name: str
    body_id: int = -1
    mass: float = 1.0
    position: List[float] = field(default_factory=lambda: [0, 0, 0])
    orientation: List[float] = field(default_factory=lambda: [0, 0, 0, 1])
    color: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    is_static: bool = False


class ShapeFactory:
    @staticmethod
    def create(shape: str, size: List[float], color: List[float]) -> Tuple[int, int]:
        import pybullet as p
        
        creators = {
            "box": ShapeFactory._create_box,
            "sphere": ShapeFactory._create_sphere,
            "cylinder": ShapeFactory._create_cylinder,
        }
        
        creator = creators.get(shape)
        if not creator:
            raise ValueError(f"Unknown shape: {shape}")
        return creator(size, color)
    
    @staticmethod
    def _create_box(size: List[float], color: List[float]) -> Tuple[int, int]:
        import pybullet as p
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        return collision, visual
    
    @staticmethod
    def _create_sphere(size: List[float], color: List[float]) -> Tuple[int, int]:
        import pybullet as p
        radius = size[0] if isinstance(size, list) else size
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
        return collision, visual
    
    @staticmethod
    def _create_cylinder(size: List[float], color: List[float]) -> Tuple[int, int]:
        import pybullet as p
        radius = size[0]
        height = size[1] if len(size) > 1 else size[0] * 2
        collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
        return collision, visual


class Scene:
    def __init__(self, gravity: float = -9.81):
        self.gravity = gravity
        self.client = None
        self.objects: Dict[str, PhysicsObject] = {}
        self.ground_id = None
        self._is_setup = False
    
    def setup(self, use_gui: bool = False) -> bool:
        import pybullet as p
        import pybullet_data

        try:
            mode = p.GUI if use_gui else p.DIRECT
            self.client = p.connect(mode)
            
            if self.client < 0:
                log.error("PyBullet connection failed")
                return False
            
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
        import pybullet as p
        self.ground_id = p.loadURDF("plane.urdf")
        return self.ground_id
    
    def add_primitive(self, name: str, shape: str = "box", size: List[float] = None,
                     mass: float = 1.0, position: List[float] = None,
                     color: List[float] = None, is_static: bool = False) -> PhysicsObject:
        import pybullet as p
        
        size = size or [0.1, 0.1, 0.1]
        position = position or [0, 0, 0.5]
        color = color or [0.5, 0.5, 0.5, 1.0]
        
        collision, visual = ShapeFactory.create(shape, size, color)
        
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
                  position: List[float] = None, scale: float = 1.0,
                  is_static: bool = False) -> Optional[PhysicsObject]:
        import pybullet as p
        
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
    
    def get_object(self, name: str) -> Optional[PhysicsObject]:
        return self.objects.get(name)
    
    def get_object_state(self, name: str) -> Tuple[List[float], List[float]]:
        import pybullet as p
        obj = self.objects.get(name)
        if obj is None:
            return [0, 0, 0], [0, 0, 0, 1]
        pos, orn = p.getBasePositionAndOrientation(obj.body_id)
        return list(pos), list(orn)
    
    def apply_force(self, name: str, force: List[float], position: List[float] = None) -> bool:
        import pybullet as p
        obj = self.objects.get(name)
        if obj is None:
            return False
        position = position or [0, 0, 0]
        p.applyExternalForce(obj.body_id, -1, forceObj=force, posObj=position, flags=p.WORLD_FRAME)
        return True
    
    def set_velocity(self, name: str, linear: List[float], angular: List[float] = None):
        import pybullet as p
        obj = self.objects.get(name)
        if obj is None:
            return
        angular = angular or [0, 0, 0]
        p.resetBaseVelocity(obj.body_id, linearVelocity=linear, angularVelocity=angular)
    
    def cleanup(self):
        import pybullet as p
        if self.client is not None:
            p.disconnect()
            self.client = None
            self._is_setup = False
            log.info("Scene cleaned up")

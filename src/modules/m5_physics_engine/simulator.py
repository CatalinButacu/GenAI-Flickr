"""Physics simulation with cinematic camera support."""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .scene import Scene

log = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fov: float = 60.0
    near: float = 0.1
    far: float = 100.0
    target: List[float] = field(default_factory=lambda: [0, 0, 0.3])
    distance: float = 1.5
    yaw: float = 45.0
    pitch: float = -30.0


@dataclass
class FrameData:
    timestamp: float
    rgb: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray


class EasingFunctions:
    # Hermite smoothstep: S(t) = 3tÂ² - 2tÂ³
    @staticmethod
    def smooth(t: float) -> float:
        return t * t * (3 - 2 * t)
    
    @staticmethod
    def linear(t: float) -> float:
        return t
    
    @staticmethod
    def ease_in(t: float) -> float:
        return t * t
    
    @staticmethod
    def ease_out(t: float) -> float:
        return 1 - (1 - t) ** 2
    
    # Cubic ease-in-out: 3tÂ² - 2tÂ³
    @staticmethod
    def ease_in_out(t: float) -> float:
        return 3 * t * t - 2 * t * t * t
    
    FUNCTIONS = {
        "linear": linear.__func__,
        "smooth": smooth.__func__,
        "ease_in": ease_in.__func__,
        "ease_out": ease_out.__func__,
        "ease_in_out": ease_in_out.__func__,
    }
    
    @staticmethod
    def apply(t: float, easing: str) -> float:
        t = max(0, min(1, t))
        func = EasingFunctions.FUNCTIONS.get(easing, EasingFunctions.linear)
        return func(t)


class CinematicCamera:
    def __init__(self, target: List[float] = None, distance: float = 2.0, 
                 yaw: float = 45.0, pitch: float = -25.0):
        self.base_target = target or [0, 0, 0.5]
        self.base_distance = distance
        self.base_yaw = yaw
        self.base_pitch = pitch
        
        self.current_target = list(self.base_target)
        self.current_distance = distance
        self.current_yaw = yaw
        self.current_pitch = pitch
        self.effects = []
    
    def add_orbit(self, start_yaw: float = None, end_yaw: float = None, 
                  start_time: float = 0.0, duration: float = 5.0, easing: str = "smooth"):
        self.effects.append({
            'type': 'orbit',
            'start_yaw': start_yaw if start_yaw is not None else self.base_yaw,
            'end_yaw': end_yaw if end_yaw is not None else self.base_yaw + 360,
            'start_time': start_time, 'duration': duration, 'easing': easing
        })
    
    def add_zoom(self, start_dist: float = None, end_dist: float = None, 
                 start_time: float = 0.0, duration: float = 2.0, easing: str = "smooth"):
        self.effects.append({
            'type': 'zoom',
            'start_dist': start_dist if start_dist is not None else self.base_distance,
            'end_dist': end_dist if end_dist is not None else self.base_distance * 0.5,
            'start_time': start_time, 'duration': duration, 'easing': easing
        })
    
    def add_pitch(self, start_pitch: float = None, end_pitch: float = None, 
                  start_time: float = 0.0, duration: float = 2.0, easing: str = "smooth"):
        self.effects.append({
            'type': 'pitch',
            'start_pitch': start_pitch if start_pitch is not None else self.base_pitch,
            'end_pitch': end_pitch if end_pitch is not None else -45.0,
            'start_time': start_time, 'duration': duration, 'easing': easing
        })
    
    def add_pan(self, start_target: List[float] = None, end_target: List[float] = None, 
                start_time: float = 0.0, duration: float = 2.0, easing: str = "smooth"):
        self.effects.append({
            'type': 'pan',
            'start_target': start_target if start_target is not None else list(self.base_target),
            'end_target': end_target if end_target is not None else [0, 0, 0.5],
            'start_time': start_time, 'duration': duration, 'easing': easing
        })
    
    def update(self, time: float):
        for effect in self.effects:
            progress = self._get_progress(time, effect)
            if progress < 0 or progress > 1:
                continue
            
            t = EasingFunctions.apply(progress, effect.get('easing', 'smooth'))
            self._apply_effect(effect, t)
    
    def _get_progress(self, time: float, effect: dict) -> float:
        start, duration = effect['start_time'], effect['duration']
        if duration <= 0:
            return 1.0 if time >= start else -1.0
        return (time - start) / duration
    
    def _apply_effect(self, effect: dict, t: float):
        effect_type = effect['type']
        if effect_type == 'orbit':
            self.current_yaw = self._interpolate(effect['start_yaw'], effect['end_yaw'], t)
        elif effect_type == 'zoom':
            self.current_distance = self._interpolate(effect['start_dist'], effect['end_dist'], t)
        elif effect_type == 'pitch':
            self.current_pitch = self._interpolate(effect['start_pitch'], effect['end_pitch'], t)
        elif effect_type == 'pan':
            start, end = effect['start_target'], effect['end_target']
            self.current_target = [self._interpolate(start[i], end[i], t) for i in range(3)]
    
    @staticmethod
    def _interpolate(start: float, end: float, t: float) -> float:
        return start + (end - start) * t
    
    def get_camera_params(self) -> Tuple[float, float, float, List[float]]:
        return self.current_yaw, self.current_pitch, self.current_distance, self.current_target
    
    def to_config(self) -> CameraConfig:
        return CameraConfig(target=self.current_target, distance=self.current_distance, 
                           yaw=self.current_yaw, pitch=self.current_pitch)


class Simulator:
    def __init__(self, scene: Scene, camera: CameraConfig = None):
        self.scene = scene
        self.camera = camera or CameraConfig()
        self.current_time = 0.0
        self.physics_hz = 240
    
    def step(self, dt: float = None):
        import pybullet as p
        dt = dt or (1.0 / self.physics_hz)
        p.stepSimulation()
        self.current_time += dt
    
    def render(self) -> FrameData:
        import pybullet as p
        
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera.target,
            distance=self.camera.distance,
            yaw=self.camera.yaw, pitch=self.camera.pitch,
            roll=0, upAxisIndex=2
        )
        
        aspect = self.camera.width / self.camera.height
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera.fov, aspect=aspect,
            nearVal=self.camera.near, farVal=self.camera.far
        )
        
        _, _, rgb, depth_buffer, seg = p.getCameraImage(
            width=self.camera.width, height=self.camera.height,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )
        
        rgb_array = np.array(rgb, dtype=np.uint8).reshape((self.camera.height, self.camera.width, 4))[:, :, :3]
        
        # Depth linearization: z = far*near / (far - (far-near)*depth_buffer)
        depth_buffer = np.array(depth_buffer).reshape((self.camera.height, self.camera.width))
        depth_linear = self.camera.far * self.camera.near / (self.camera.far - (self.camera.far - self.camera.near) * depth_buffer)
        depth_uint8 = (np.clip(depth_linear / 5.0, 0, 1) * 255).astype(np.uint8)
        
        seg_array = np.array(seg).reshape((self.camera.height, self.camera.width)).astype(np.int32)
        
        return FrameData(timestamp=self.current_time, rgb=rgb_array, depth=depth_uint8, segmentation=seg_array)
    
    def run(self, duration: float = 5.0, fps: int = 24, actions: List[Dict] = None) -> List[FrameData]:
        actions = actions or []
        frames = []
        total_steps = int(duration * self.physics_hz)
        render_interval = self.physics_hz // fps
        sorted_actions = sorted(actions, key=lambda a: a.get('time', 0))
        action_index = 0
        
        log.info("Simulation: %ss %sfps %d actions", duration, fps, len(actions))
        
        for step in range(total_steps):
            while action_index < len(sorted_actions):
                action = sorted_actions[action_index]
                if action.get('time', 0) <= self.current_time:
                    self._apply_action(action)
                    action_index += 1
                else:
                    break
            
            self.step()
            
            if step % render_interval == 0:
                frames.append(self.render())
        
        log.info("Captured %d frames", len(frames))
        return frames
    
    def run_cinematic(self, duration: float = 5.0, fps: int = 24, actions: List[Dict] = None,
                      cinematic_camera: CinematicCamera = None) -> List[FrameData]:
        actions = actions or []
        frames = []
        
        if cinematic_camera is None:
            cinematic_camera = CinematicCamera(
                target=self.camera.target, distance=self.camera.distance,
                yaw=self.camera.yaw, pitch=self.camera.pitch
            )
            cinematic_camera.add_orbit(start_yaw=self.camera.yaw, end_yaw=self.camera.yaw + 90, duration=duration)
        
        total_steps = int(duration * self.physics_hz)
        render_interval = self.physics_hz // fps
        sorted_actions = sorted(actions, key=lambda a: a.get('time', 0))
        action_index = 0
        
        log.info("Cinematic: %ss %sfps %d effects", duration, fps, len(cinematic_camera.effects))
        
        for step in range(total_steps):
            while action_index < len(sorted_actions):
                action = sorted_actions[action_index]
                if action.get('time', 0) <= self.current_time:
                    self._apply_action(action)
                    action_index += 1
                else:
                    break
            
            self.step()
            
            if step % render_interval == 0:
                cinematic_camera.update(self.current_time)
                yaw, pitch, dist, target = cinematic_camera.get_camera_params()
                self.camera.yaw, self.camera.pitch, self.camera.distance, self.camera.target = yaw, pitch, dist, target
                frames.append(self.render())
        
        log.info("Captured %d cinematic frames", len(frames))
        return frames
    
    def _apply_action(self, action: Dict):
        action_type = action.get('type', 'force')
        obj_name = action.get('object')
        
        if action_type == 'force':
            self.scene.apply_force(obj_name, action.get('force', [0, 0, 0]))
        elif action_type == 'velocity':
            self.scene.set_velocity(obj_name, action.get('velocity', [0, 0, 0]))
    
    def save_frames(self, frames: List[FrameData], output_dir: str, 
                    save_depth: bool = True, save_segmentation: bool = False) -> Dict[str, List[str]]:
        from PIL import Image
        os.makedirs(output_dir, exist_ok=True)
        paths = {'rgb': [], 'depth': [], 'segmentation': []}
        
        for i, frame in enumerate(frames):
            rgb_path = os.path.join(output_dir, f'frame_{i:04d}_rgb.png')
            Image.fromarray(frame.rgb).save(rgb_path)
            paths['rgb'].append(rgb_path)
            
            if save_depth:
                depth_path = os.path.join(output_dir, f'frame_{i:04d}_depth.png')
                Image.fromarray(frame.depth).save(depth_path)
                paths['depth'].append(depth_path)
            
            if save_segmentation:
                seg_path = os.path.join(output_dir, f'frame_{i:04d}_seg.png')
                seg_vis = ((frame.segmentation + 1) * 30 % 256).astype(np.uint8)
                Image.fromarray(seg_vis).save(seg_path)
                paths['segmentation'].append(seg_path)
        
        log.info("Saved %d frames", len(frames))
        return paths
    
    def reset(self):
        self.current_time = 0.0
    
    def create_video(self, frames: List[FrameData], output_path: str, 
                     fps: int = 24, layout: str = "horizontal") -> str:
        import imageio
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
        
        segment_colors = np.array([
            [0,0,0], [255,0,0], [0,255,0], [0,0,255], 
            [255,255,0], [255,0,255], [0,255,255], [255,128,0], [128,0,255]
        ], dtype=np.uint8)
        
        for frame in frames:
            depth_rgb = np.stack([frame.depth] * 3, axis=-1)
            seg_ids = (frame.segmentation + 1) % len(segment_colors)
            seg_colored = segment_colors[seg_ids]

            if layout == "rgb":
                combined = frame.rgb
            elif layout == "horizontal":
                combined = np.concatenate([frame.rgb, depth_rgb, seg_colored], axis=1)
            elif layout == "vertical":
                combined = np.concatenate([frame.rgb, depth_rgb, seg_colored], axis=0)
            else:
                from PIL import Image
                h, w = frame.rgb.shape[:2]
                bottom = np.concatenate([depth_rgb, seg_colored], axis=1)
                bottom_resized = np.array(Image.fromarray(bottom).resize((w, h // 2)))
                combined = np.concatenate([frame.rgb, bottom_resized], axis=0)
            
            writer.append_data(combined)
        
        writer.close()
        log.info("Video created: %s", output_path)
        return output_path


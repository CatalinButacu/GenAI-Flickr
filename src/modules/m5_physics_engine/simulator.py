"""
#WHERE
    Used by pipeline.py, demo scripts, benchmark_m5.py, and test_physics_engine.py.

#WHAT
    PyBullet physics simulation runner — steps physics, renders RGB/depth/seg
    frames, supports cinematic camera effects, and writes video files.

#INPUT
    Scene (loaded 3D objects), CameraConfig, duration, fps, actions list.

#OUTPUT
    List[FrameData] with per-frame RGB, depth, and segmentation arrays;
    optionally an MP4 video file.
"""

import os
import logging
from typing import Dict, List

import numpy as np

from .camera import (                            # noqa: F401 — re-exports
    CameraConfig, CinematicCamera, EasingFunctions, FrameData,
)
from .scene import Scene
from src.shared.mem_profile import profile_memory

log = logging.getLogger(__name__)


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
    
    @profile_memory
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
    
    @profile_memory
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


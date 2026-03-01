#!/usr/bin/env python3
"""Demo: Cinematic Camera Motion (orbit, zoom, pitch)"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.physics_engine import Scene, Simulator, CameraConfig, CinematicCamera

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Demo: Cinematic Camera Motion")
    logger.info("=" * 60)
    
    # Setup scene
    scene = Scene(gravity=-9.81)
    scene.setup()
    scene.add_ground()
    
    scene.add_primitive(name="red_ball", shape="sphere", size=[0.15], mass=1.0, position=[0, 0, 1.5], color=[0.9, 0.1, 0.1, 1.0])
    scene.add_primitive(name="blue_box", shape="box", size=[0.2, 0.2, 0.2], mass=2.0, position=[0.3, 0, 0.5], color=[0.1, 0.1, 0.9, 1.0])
    scene.add_primitive(name="green_cylinder", shape="cylinder", size=[0.1, 0.3], mass=0.5, position=[-0.3, 0, 0.8], color=[0.1, 0.9, 0.1, 1.0])
    scene.add_primitive(name="platform", shape="box", size=[0.5, 0.5, 0.05], mass=0, position=[0, 0, 0.2], color=[0.4, 0.3, 0.2, 1.0], is_static=True)
    
    # Configure cinematic camera
    cinematic = CinematicCamera(target=[0, 0, 0.5], distance=3.0, yaw=0, pitch=-15)
    cinematic.add_orbit(start_yaw=0, end_yaw=360, duration=6.0, easing="smooth")
    cinematic.add_zoom(start_dist=3.0, end_dist=1.8, start_time=1.0, duration=2.0, easing="ease_out")
    cinematic.add_zoom(start_dist=1.8, end_dist=2.5, start_time=4.0, duration=2.0, easing="ease_in")
    cinematic.add_pitch(start_pitch=-15, end_pitch=-35, start_time=2.0, duration=2.0, easing="smooth")
    
    # Simulate
    camera = CameraConfig(width=640, height=480, target=[0, 0, 0.5], distance=2.5, yaw=0, pitch=-20)
    sim = Simulator(scene, camera)
    
    actions = [
        {"time": 1.0, "object": "red_ball", "type": "force", "force": [3, 2, 0]},
        {"time": 2.0, "object": "green_cylinder", "type": "velocity", "velocity": [1, 0, 2]},
    ]
    
    logger.info("Running cinematic simulation...")
    frames = sim.run_cinematic(duration=6.0, fps=24, actions=actions, cinematic_camera=cinematic)
    
    # Create videos
    logger.info("Creating videos...")
    sim.create_video(frames=frames, output_path="outputs/videos/cinematic_demo.mp4", fps=24, layout="horizontal")
    
    import imageio
    rgb_path = "outputs/videos/cinematic_rgb_only.mp4"
    os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
    writer = imageio.get_writer(rgb_path, fps=24, codec='libx264')
    for frame in frames:
        writer.append_data(frame.rgb)
    writer.close()
    logger.info(f"Created RGB-only video: {rgb_path}")
    
    scene.cleanup()
    
    logger.info("\n" + "=" * 60)
    logger.info("Cinematic Demo Complete!")
    logger.info(f"RGB-only video: {rgb_path}")
    logger.info("=" * 60)
    print(f"\n🎬 Open: {rgb_path}")


if __name__ == "__main__":
    main()

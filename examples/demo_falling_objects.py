#!/usr/bin/env python3
"""Demo: Falling Objects Physics Simulation"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.physics_engine import Scene, Simulator, CameraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Demo: Falling Objects with Physics Simulation")
    logger.info("=" * 60)
    
    # Setup scene
    scene = Scene(gravity=-9.81)
    scene.setup()
    scene.add_ground()
    
    scene.add_primitive(name="ball", shape="sphere", size=[0.1], mass=1.0, position=[0, 0, 1.5], color=[1.0, 0.2, 0.2, 1.0])
    scene.add_primitive(name="box", shape="box", size=[0.15, 0.15, 0.15], mass=2.0, position=[0.3, 0, 1.0], color=[0.2, 0.6, 1.0, 1.0])
    scene.add_primitive(name="cylinder", shape="cylinder", size=[0.08, 0.25], mass=0.5, position=[-0.2, 0, 1.2], color=[0.2, 1.0, 0.3, 1.0])
    scene.add_primitive(name="platform", shape="box", size=[0.3, 0.3, 0.02], mass=0, position=[0, 0, 0.3], color=[0.4, 0.3, 0.2, 1.0], is_static=True)
    
    # Setup camera and simulator
    camera = CameraConfig(width=640, height=480, target=[0, 0, 0.5], distance=2.0, yaw=30, pitch=-25)
    sim = Simulator(scene, camera)
    
    actions = [
        {"time": 0.5, "object": "ball", "type": "force", "force": [5, 0, 0]},
        {"time": 1.5, "object": "box", "type": "velocity", "velocity": [-1, 0, 2]},
    ]
    
    # Run simulation
    logger.info("Running simulation...")
    frames = sim.run(duration=4.0, fps=24, actions=actions)
    
    # Create video
    os.makedirs("outputs/videos", exist_ok=True)
    video_path = sim.create_video(frames=frames, output_path="outputs/videos/falling_objects.mp4", fps=24, layout="horizontal")
    
    scene.cleanup()
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info(f"Video: {video_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

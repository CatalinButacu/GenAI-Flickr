#!/usr/bin/env python3
"""Demo: Text to 3D Model to Physics Simulation to Video"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.asset_generator import ModelGenerator
from src.modules.physics_engine import Scene, Simulator, CameraConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Demo: Text → 3D Model → Physics Simulation → Video")
    logger.info("=" * 60)
    
    prompt = "a BMW i3 car going in circle"
    output_dir = "assets/objects"
    
    # Step 1: Generate 3D Model
    logger.info("\n[Step 1] Generating 3D model from text...")
    generator = ModelGenerator(backend="shap-e")
    
    if not generator.setup():
        logger.warning("Shap-E not available. Using placeholder cube.")
        mesh_path = None
    else:
        result = generator.generate_from_text(prompt=prompt, output_dir=output_dir, name="car")
        mesh_path = result.mesh_path if result else None
    
    # Step 2: Physics Scene
    logger.info("\n[Step 2] Setting up physics scene...")
    scene = Scene(gravity=-9.81)
    scene.setup()
    scene.add_ground()
    
    if mesh_path and os.path.exists(mesh_path):
        logger.info(f"Loading generated mesh: {mesh_path}")
        scene.load_mesh(name="car", mesh_path=mesh_path, mass=5.0, position=[0, 0, 1.0], scale=0.5)
    else:
        scene.add_primitive(name="car", shape="box", size=[0.2, 0.2, 0.3], mass=5.0, position=[0, 0, 1.0], color=[0.6, 0.4, 0.2, 1.0])
    
    scene.add_primitive(name="ball", shape="sphere", size=[0.1], mass=0.5, position=[0.3, 0, 2.0], color=[0.9, 0.1, 0.1, 1.0])
    
    # Step 3: Simulate
    logger.info("\n[Step 3] Running physics simulation...")
    camera = CameraConfig(width=640, height=480, target=[0, 0, 0.5], distance=2.5, yaw=30, pitch=-20)
    sim = Simulator(scene, camera)
    
    actions = [{"time": 0.3, "object": "ball", "type": "velocity", "velocity": [-2, 0, 0]}]
    frames = sim.run(duration=3.0, fps=24, actions=actions)
    
    # Step 4: Video
    logger.info("\n[Step 4] Creating video...")
    video_path = sim.create_video(frames=frames, output_path="outputs/videos/text_to_physics.mp4", fps=24, layout="horizontal")
    
    scene.cleanup()
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo complete!")
    logger.info(f"Generated model: {mesh_path or 'placeholder'}")
    logger.info(f"Video: {video_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

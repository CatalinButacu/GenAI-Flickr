"""
Unit Tests for Pipeline Components
==================================
Comprehensive tests for all 8 modules of the physics-constrained video pipeline.

Run with: pytest tests/test_modules.py -v
Or: python tests/test_modules.py
"""

import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSharedVocabulary(unittest.TestCase):
    """Tests for the shared vocabulary system."""
    
    def test_actions_exist(self):
        """Vocabulary contains expected action categories."""
        from src.shared.vocabulary import ACTIONS, ActionCategory
        
        self.assertGreater(len(ACTIONS), 0)
        
        # Check key actions exist
        self.assertIn("walk", ACTIONS)
        self.assertIn("kick", ACTIONS)
        self.assertIn("fall", ACTIONS)
        
    def test_objects_exist(self):
        """Vocabulary contains expected objects."""
        from src.shared.vocabulary import OBJECTS
        
        # Objects use canonical names (sphere, cube) not keywords (ball)
        self.assertIn("sphere", OBJECTS)  # 'ball' is a keyword alias for 'sphere'
        self.assertIn("cube", OBJECTS)
        self.assertIn("humanoid", OBJECTS)
        
    def test_action_lookup(self):
        """Action lookup by keyword works."""
        from src.shared.vocabulary import get_action_by_keyword
        
        action = get_action_by_keyword("walks")
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "walk")
        
        action = get_action_by_keyword("kicks")
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "kick")
        
    def test_physics_actions(self):
        """Physics actions (fall, roll, etc.) are defined."""
        from src.shared.vocabulary import ACTIONS, ActionCategory
        
        physics_actions = [a for a in ACTIONS.values() 
                         if a.category == ActionCategory.PHYSICS]
        self.assertGreater(len(physics_actions), 0)
        
        # Check fall action properties
        fall = ACTIONS.get("fall")
        self.assertIsNotNone(fall)
        self.assertTrue(fall.requires_target)


class TestPromptParser(unittest.TestCase):
    """Tests for Module 1: Prompt Parser."""
    
    def setUp(self):
        from src.modules.m1_scene_understanding.prompt_parser import PromptParser
        self.parser = PromptParser(mode="rules")
        self.parser.setup()
        
    def test_simple_parse(self):
        """Parse simple prompt with one entity."""
        result = self.parser.parse("A red ball")
        
        self.assertEqual(len(result.entities), 1)
        # Parser may normalize 'ball' to 'sphere' based on vocabulary
        self.assertIn(result.entities[0].object_type, ["ball", "sphere"])
        
    def test_action_parse(self):
        """Parse prompt with action."""
        result = self.parser.parse("A person walks forward")
        
        self.assertGreater(len(result.actions), 0)
        self.assertEqual(result.actions[0].action_type, "walk")
        
    def test_fall_action(self):
        """Parse physics action (fall)."""
        result = self.parser.parse("A ball falls on a cube")
        
        actions = [a.action_type for a in result.actions]
        self.assertIn("fall", actions)
        
    def test_multiple_entities(self):
        """Parse prompt with multiple entities."""
        result = self.parser.parse("A red ball and a blue cube")
        
        self.assertGreaterEqual(len(result.entities), 2)


class TestScenePlanner(unittest.TestCase):
    """Tests for Module 2: Scene Planner."""
    
    def setUp(self):
        from src.modules.m1_scene_understanding.prompt_parser import PromptParser
        from src.modules.m2_scene_planner import ScenePlanner
        
        self.parser = PromptParser(mode="rules")
        self.parser.setup()
        self.planner = ScenePlanner()
        
    def test_basic_positioning(self):
        """Entities are assigned 3D positions."""
        parsed = self.parser.parse("A ball and a cube")
        planned = self.planner.plan(parsed)
        
        self.assertGreater(len(planned.entities), 0)
        
        for entity in planned.entities:
            self.assertIsNotNone(entity.position)
            self.assertIsNotNone(entity.position.x)
            self.assertIsNotNone(entity.position.y)
            self.assertIsNotNone(entity.position.z)
            
    def test_fall_positioning(self):
        """Falling objects are positioned above targets."""
        parsed = self.parser.parse("A ball falls on a cube")
        planned = self.planner.plan(parsed)
        
        # At least one object should have z > 1 (elevated for falling)
        max_z = max(e.position.z for e in planned.entities)
        self.assertGreater(max_z, 0.5)


class TestMotionGenerator(unittest.TestCase):
    """Tests for Module 4: Motion Generator."""
    
    def test_placeholder_generation(self):
        """Placeholder motion generator creates frames."""
        from src.modules.m4_motion_generator import MotionGenerator
        
        gen = MotionGenerator(backend="placeholder")
        gen.setup()
        
        clip = gen.generate("walk forward", duration=2.0, fps=30)
        
        self.assertEqual(len(clip.frames), 60)  # 2s * 30fps
        self.assertEqual(clip.fps, 30)
        self.assertAlmostEqual(clip.duration, 2.0, places=1)
        
    def test_motion_frame_structure(self):
        """Motion frames have correct structure."""
        from src.modules.m4_motion_generator import MotionGenerator
        
        gen = MotionGenerator()
        gen.setup()
        clip = gen.generate("walk", duration=1.0)
        
        frame = clip.frames[0]
        self.assertIsNotNone(frame.timestamp)
        self.assertIsNotNone(frame.root_position)
        self.assertEqual(len(frame.root_position), 3)


class TestSSMMotionGenerator(unittest.TestCase):
    """Tests for SSM-enhanced Motion Generator."""
    
    def test_ssm_generation(self):
        """SSM motion generator creates frames."""
        from src.modules.m4_motion_generator import SSMMotionGenerator
        
        gen = SSMMotionGenerator(backend="ssm")
        gen.setup()
        
        clip = gen.generate("walk", duration=1.0, fps=30)
        
        self.assertEqual(len(clip.frames), 30)
        
    def test_ssm_fallback(self):
        """SSM falls back gracefully if torch unavailable."""
        from src.modules.m4_motion_generator import SSMMotionGenerator
        
        gen = SSMMotionGenerator(backend="ssm")
        success = gen.setup()
        
        self.assertTrue(success)


class TestSSMCore(unittest.TestCase):
    """Tests for SSM module core components."""
    
    def test_ssm_info(self):
        """SSM info returns expected structure."""
        from src.ssm import get_ssm_info
        
        info = get_ssm_info()
        
        self.assertIn("torch_available", info)
        self.assertIn("layers", info)
        self.assertIn("references", info)
        self.assertIn("novel_contribution", info)
        
    def test_numpy_ssm(self):
        """NumPy SSM fallback works."""
        from src.ssm import SimpleSSMNumpy
        
        ssm = SimpleSSMNumpy(d_state=8, d_input=16, d_output=16)
        
        # Test single step
        u = np.random.default_rng(seed=42).standard_normal(16)
        y = ssm.step(u)
        
        self.assertEqual(len(y), 16)
        
        # Test sequence
        ssm.reset()
        seq = np.random.default_rng(seed=42).standard_normal((10, 16))
        out = ssm.forward(seq)
        
        self.assertEqual(out.shape, (10, 16))


class TestPhysicsEngine(unittest.TestCase):
    """Tests for Module 5: Physics Engine."""
    
    def test_scene_creation(self):
        """Physics scene can be created and cleaned up."""
        from src.modules.m5_physics_engine import Scene
        
        scene = Scene(gravity=-9.81)
        scene.setup()
        
        self.assertTrue(scene._is_setup)
        
        scene.cleanup()
        
    def test_add_primitives(self):
        """Can add primitive shapes to scene."""
        from src.modules.m5_physics_engine import Scene
        
        scene = Scene()
        scene.setup()
        scene.add_ground()
        
        # Add various primitives
        scene.add_primitive("ball", "sphere", 0.1, 1.0, [0, 0, 1])
        scene.add_primitive("box", "box", 0.2, 2.0, [0.5, 0, 0.5])
        
        scene.cleanup()


class TestHumanoidBody(unittest.TestCase):
    """Tests for Humanoid Body Loader."""
    
    def test_import(self):
        """HumanoidBody can be imported."""
        from src.modules.m5_physics_engine.humanoid import HumanoidBody, HumanoidConfig
        
        config = HumanoidConfig(height=1.7, mass=70.0)
        body = HumanoidBody(config)
        
        self.assertEqual(body.config.height, 1.7)
        
    def test_joint_indices(self):
        """Joint indices are defined."""
        from src.modules.m5_physics_engine.humanoid import HumanoidBody
        
        self.assertIn("left_knee", HumanoidBody.JOINT_INDICES)
        self.assertIn("right_knee", HumanoidBody.JOINT_INDICES)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_pipeline_setup(self):
        """Pipeline can be set up with all modules."""
        from src.pipeline_v2 import Pipeline, PipelineConfig
        
        config = PipelineConfig(
            use_asset_generation=False,
            use_motion_generation=True,
            use_ai_enhancement=False
        )
        
        pipeline = Pipeline(config)
        success = pipeline.setup()
        
        self.assertTrue(success)
        self.assertTrue(pipeline._is_setup)
        
    def test_pipeline_run(self):
        """Pipeline can process a prompt end-to-end."""
        from src.pipeline_v2 import Pipeline, PipelineConfig
        
        config = PipelineConfig(
            use_asset_generation=False,
            use_motion_generation=True,
            use_ai_enhancement=False,
            duration=1.0,
            fps=12
        )
        
        pipeline = Pipeline(config)
        result = pipeline.run("A ball falls", output_name="test_integration")
        
        self.assertIn("parsed_scene", result)
        self.assertIn("physics_frames", result)
        self.assertIn("video_path", result)
        
        self.assertGreater(len(result["physics_frames"]), 0)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)

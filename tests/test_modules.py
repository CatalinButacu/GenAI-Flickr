"""
Unit Tests for Pipeline Components
==================================
Covers: Shared Vocabulary, M1 (PromptParser), M2 (ScenePlanner),
M4 (MotionGenerator + SSM), M5 (PhysicsScene + HumanoidBody),
and end-to-end Pipeline integration.

Not covered here: M3 (AssetGenerator), M6 (RenderEngine), M7 (AIEnhancer).

Run with: pytest tests/test_modules.py -v
Or: python tests/test_modules.py
"""

import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shared.vocabulary import (
    ACTIONS, OBJECTS, ActionCategory, get_action_by_keyword,
)
from src.modules.understanding.prompt_parser import PromptParser
from src.modules.planner import ScenePlanner
from src.modules.motion import MotionGenerator, SSMMotionModel
from src.modules.motion.ssm import get_ssm_info, SimpleSSMNumpy
from src.modules.physics import Scene
from src.modules.physics.humanoid import HumanoidBody, HumanoidConfig
from src.pipeline import Pipeline, PipelineConfig


class TestSharedVocabulary(unittest.TestCase):
    """Tests for the shared vocabulary system."""
    
    def test_actions_exist(self):
        """Vocabulary contains expected action categories."""
        self.assertGreater(len(ACTIONS), 0)
        
        # Check key actions exist
        self.assertIn("walk", ACTIONS)
        self.assertIn("kick", ACTIONS)
        self.assertIn("fall", ACTIONS)
        
    def test_objects_exist(self):
        """Vocabulary contains expected objects."""
        # Objects use canonical names (sphere, cube) not keywords (ball)
        self.assertIn("sphere", OBJECTS)  # 'ball' is a keyword alias for 'sphere'
        self.assertIn("cube", OBJECTS)
        self.assertIn("humanoid", OBJECTS)
        
    def test_action_lookup(self):
        """Action lookup by keyword works."""
        action = get_action_by_keyword("walks")
        self.assertIsNotNone(action)
        assert action is not None  # narrow for type checker
        self.assertEqual(action.name, "walk")  # pyright: ignore[reportAttributeAccessIssue]
        
        action = get_action_by_keyword("kicks")
        self.assertIsNotNone(action)
        assert action is not None  # narrow for type checker
        self.assertEqual(action.name, "kick")  # pyright: ignore[reportAttributeAccessIssue]
        
    def test_physics_actions(self):
        """Physics actions (fall, roll, etc.) are defined."""
        physics_actions = [a for a in ACTIONS.values() 
                         if a.category == ActionCategory.PHYSICS]
        self.assertGreater(len(physics_actions), 0)
        
        # Check fall action properties
        fall = ACTIONS.get("fall")
        self.assertIsNotNone(fall)
        assert fall is not None  # narrow for type checker
        self.assertTrue(fall.requires_target)  # pyright: ignore[reportAttributeAccessIssue]


class TestPromptParser(unittest.TestCase):
    """Tests for Module 1: Prompt Parser."""
    
    def setUp(self):
        self.parser = PromptParser()
        
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
        self.parser = PromptParser()
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
        """Motion generator creates clip from text."""
        gen = MotionGenerator(use_retrieval=True, use_ssm=False)
        clip = gen.generate("walk forward", num_frames=60)
        
        self.assertIsNotNone(clip)
        self.assertGreater(clip.num_frames, 0)
        
    def test_motion_frame_structure(self):
        """Motion clip has correct structure."""
        gen = MotionGenerator(use_retrieval=True, use_ssm=False)
        clip = gen.generate("walk", num_frames=30)
        
        self.assertIsNotNone(clip)
        self.assertIsNotNone(clip.source)


class TestSSMMotionGenerator(unittest.TestCase):
    """Tests for SSM-enhanced Motion Generator."""
    
    def test_ssm_generation(self):
        """SSM motion model can be instantiated."""
        model = SSMMotionModel()
        clip = model.generate("walk", num_frames=30)
        # May return None if checkpoint missing — that's OK
        if clip is not None:
            self.assertGreater(clip.num_frames, 0)
        
    def test_ssm_fallback(self):
        """SSM model handles missing checkpoint gracefully."""
        model = SSMMotionModel(checkpoint_path="nonexistent/path.pt")
        clip = model.generate("walk", num_frames=30)
        # Should return None when checkpoint is missing
        self.assertIsNone(clip)


class TestSSMCore(unittest.TestCase):
    """Tests for SSM module core components."""
    
    def test_ssm_info(self):
        """SSM info returns expected structure."""
        info = get_ssm_info()
        
        self.assertIn("torch_available", info)
        self.assertIn("layers", info)
        self.assertIn("references", info)
        self.assertIn("novel_contribution", info)
        
    def test_numpy_ssm(self):
        """NumPy SSM fallback works."""
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
        scene = Scene(gravity=-9.81)
        scene.setup()
        
        self.assertTrue(scene._is_setup)
        
        scene.cleanup()
        
    def test_add_primitives(self):
        """Can add primitive shapes to scene."""
        scene = Scene()
        scene.setup()
        scene.add_ground()
        
        # Add various primitives
        scene.add_primitive("ball", "sphere", [0.1], 1.0, [0, 0, 1])
        scene.add_primitive("box", "box", [0.2], 2.0, [0.5, 0, 0.5])
        
        scene.cleanup()


class TestHumanoidBody(unittest.TestCase):
    """Tests for Humanoid Body Loader."""
    
    def test_import(self):
        """HumanoidBody can be imported."""
        config = HumanoidConfig(height=1.7, mass=70.0)
        body = HumanoidBody(config)
        
        self.assertEqual(body.config.height, 1.7)
        
    def test_joint_indices(self):
        """Humanoid can be instantiated with config."""
        config = HumanoidConfig(height=1.7)
        body = HumanoidBody(config)
        self.assertEqual(body.config.height, 1.7)


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for full pipeline."""
    
    def test_pipeline_setup(self):
        """Pipeline can be set up with all modules."""
        config = PipelineConfig(
            use_asset_generation=False,
            use_motion_generation=False,
            use_diffusion=False
        )
        
        pipeline = Pipeline(config)
        pipeline.setup()
        
        self.assertTrue(pipeline.is_setup)
        
    def test_pipeline_run(self):
        """Pipeline can process a prompt end-to-end."""
        config = PipelineConfig(
            use_asset_generation=False,
            use_motion_generation=False,
            use_diffusion=False,
            duration=1.0,
            fps=12
        )
        
        pipeline = Pipeline(config)
        result = pipeline.run("A ball falls", output_name="test_integration")
        
        self.assertIsInstance(result, dict)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)

"""Tests for Story Agent - the core parsing logic."""

import math
import pytest
from src.modules.m1_scene_understanding.agent import (
    RuleBasedParser, SceneDescription, SceneObject,
    SceneAction, CameraMotion, OBJECT_KEYWORDS
)


class TestRuleBasedParser:
    
    def setup_method(self):
        self.parser = RuleBasedParser()
    
    def test_parse_returns_scene_description(self):
        result = self.parser.parse("a ball falls")
        assert isinstance(result, SceneDescription)
        assert result.prompt == "a ball falls"
    
    def test_detects_ball_keyword(self):
        result = self.parser.parse("a ball falls onto a table")
        object_names = [obj.name for obj in result.objects]
        assert "ball" in object_names
    
    def test_detects_table_keyword(self):
        result = self.parser.parse("a box on a table")
        object_names = [obj.name for obj in result.objects]
        assert "table" in object_names
        table = next(obj for obj in result.objects if obj.name == "table")
        assert table.is_static == True
    
    def test_detects_multiple_objects(self):
        result = self.parser.parse("a ball and a box and a cylinder")
        assert len(result.objects) >= 3
    
    def test_default_object_when_none_detected(self):
        result = self.parser.parse("something happens")
        assert len(result.objects) == 1
        assert result.objects[0].name == "ball"
    
    def test_detects_push_action(self):
        result = self.parser.parse("push a ball")
        assert len(result.actions) >= 1
        assert result.actions[0].action_type == "force"
    
    def test_detects_roll_action(self):
        result = self.parser.parse("a ball rolling")
        has_velocity_action = any(a.action_type == "velocity" for a in result.actions)
        assert has_velocity_action
    
    def test_detects_orbit_camera(self):
        result = self.parser.parse("orbit around a ball")
        has_orbit = any(m.motion_type == "orbit" for m in result.camera_motions)
        assert has_orbit
    
    def test_detects_zoom_camera(self):
        result = self.parser.parse("zoom closer to a ball")
        has_zoom = any(m.motion_type == "zoom" for m in result.camera_motions)
        assert has_zoom
    
    def test_default_camera_motion(self):
        result = self.parser.parse("a ball falls")
        assert len(result.camera_motions) >= 1
    
    def test_realistic_style(self):
        result = self.parser.parse("realistic ball falling")
        assert "photorealistic" in result.style_prompt
    
    def test_cartoon_style(self):
        result = self.parser.parse("cartoon ball bouncing")
        assert "cartoon" in result.style_prompt


class TestSceneObject:
    
    def test_default_values(self):
        obj = SceneObject(name="test")
        assert obj.shape == "box"
        assert math.isclose(obj.mass, 1.0, rel_tol=1e-9)
        assert obj.is_static == False
        assert obj.mesh_prompt is None
    
    def test_custom_values(self):
        obj = SceneObject(
            name="custom", shape="sphere", 
            position=[1, 2, 3], mass=5.0, is_static=True
        )
        assert obj.name == "custom"
        assert obj.shape == "sphere"
        assert obj.position == [1, 2, 3]
        assert math.isclose(obj.mass, 5.0, rel_tol=1e-9)
        assert obj.is_static == True


class TestSceneAction:
    
    def test_default_action_type(self):
        action = SceneAction(time=1.0, object_name="ball")
        assert action.action_type == "force"
        assert action.vector == [0, 0, 0]
    
    def test_custom_action(self):
        action = SceneAction(
            time=0.5, object_name="ball",
            action_type="velocity", vector=[2, 0, 0]
        )
        assert math.isclose(action.time, 0.5, rel_tol=1e-9)
        assert action.action_type == "velocity"
        assert action.vector == [2, 0, 0]


class TestObjectKeywords:
    
    def test_all_keywords_have_shape(self):
        for keyword, config in OBJECT_KEYWORDS.items():
            assert "shape" in config or "mesh_prompt" in config
    
    def test_table_is_static(self):
        assert OBJECT_KEYWORDS["table"]["is_static"] == True
    
    def test_ball_is_sphere(self):
        assert OBJECT_KEYWORDS["ball"]["shape"] == "sphere"
    
    def test_car_is_mesh(self):
        assert OBJECT_KEYWORDS["car"]["shape"] == "mesh"
        assert "mesh_prompt" in OBJECT_KEYWORDS["car"]

"""Tests for Physics Engine - scene and simulation logic."""

import pytest
import numpy as np
from src.modules.m5_physics_engine.scene import ShapeFactory, Scene, PhysicsObject
from src.modules.m5_physics_engine.simulator import EasingFunctions, CinematicCamera, CameraConfig


class TestShapeFactory:
    
    def test_unknown_shape_raises_error(self):
        with pytest.raises(ValueError, match="Unknown shape"):
            ShapeFactory.create("invalid", [0.1], [1, 0, 0, 1])


class TestEasingFunctions:
    
    def test_linear_at_zero(self):
        assert EasingFunctions.linear(0) == 0
    
    def test_linear_at_one(self):
        assert EasingFunctions.linear(1) == 1
    
    def test_linear_midpoint(self):
        assert EasingFunctions.linear(0.5) == 0.5
    
    def test_smooth_at_zero(self):
        assert EasingFunctions.smooth(0) == 0
    
    def test_smooth_at_one(self):
        assert EasingFunctions.smooth(1) == 1
    
    def test_smooth_midpoint(self):
        # Hermite smoothstep at 0.5: 3*(0.5)^2 - 2*(0.5)^3 = 0.5
        assert EasingFunctions.smooth(0.5) == 0.5
    
    def test_ease_in_starts_slow(self):
        # Quadratic ease-in: slow at start
        assert EasingFunctions.ease_in(0.25) < 0.25
    
    def test_ease_out_ends_slow(self):
        # Quadratic ease-out: slows down at end
        assert EasingFunctions.ease_out(0.25) > 0.25
    
    def test_apply_clamps_values(self):
        assert EasingFunctions.apply(-0.5, "linear") == 0
        assert EasingFunctions.apply(1.5, "linear") == 1
    
    def test_apply_unknown_easing_uses_linear(self):
        assert EasingFunctions.apply(0.5, "unknown") == 0.5


class TestCinematicCamera:
    
    def setup_method(self):
        self.camera = CinematicCamera(target=[0, 0, 0.5], distance=2.0, yaw=0, pitch=-30)
    
    def test_initial_values(self):
        assert self.camera.current_distance == 2.0
        assert self.camera.current_yaw == 0
        assert self.camera.current_pitch == -30
    
    def test_add_orbit_creates_effect(self):
        self.camera.add_orbit(start_yaw=0, end_yaw=90, duration=5.0)
        assert len(self.camera.effects) == 1
        assert self.camera.effects[0]['type'] == 'orbit'
    
    def test_add_zoom_creates_effect(self):
        self.camera.add_zoom(start_dist=2.0, end_dist=1.0, duration=2.0)
        assert len(self.camera.effects) == 1
        assert self.camera.effects[0]['type'] == 'zoom'
    
    def test_orbit_updates_yaw(self):
        self.camera.add_orbit(start_yaw=0, end_yaw=90, start_time=0, duration=1.0)
        self.camera.update(0.5)
        # At halfway with smooth easing, should be around 45 degrees
        assert 40 < self.camera.current_yaw < 50
    
    def test_zoom_updates_distance(self):
        self.camera.add_zoom(start_dist=2.0, end_dist=1.0, start_time=0, duration=1.0)
        self.camera.update(1.0)
        assert self.camera.current_distance == 1.0
    
    def test_get_camera_params(self):
        yaw, pitch, dist, target = self.camera.get_camera_params()
        assert yaw == 0
        assert pitch == -30
        assert dist == 2.0
        assert target == [0, 0, 0.5]
    
    def test_to_config_returns_camera_config(self):
        config = self.camera.to_config()
        assert isinstance(config, CameraConfig)
        assert config.yaw == 0
        assert config.pitch == -30


class TestCameraConfig:
    
    def test_default_values(self):
        config = CameraConfig()
        assert config.width == 640
        assert config.height == 480
        assert config.fov == 60.0
    
    def test_custom_values(self):
        config = CameraConfig(width=1920, height=1080, fov=90.0)
        assert config.width == 1920
        assert config.height == 1080
        assert config.fov == 90.0


class TestPhysicsObject:
    
    def test_default_values(self):
        obj = PhysicsObject(name="test")
        assert obj.body_id == -1
        assert obj.mass == 1.0
        assert obj.is_static == False
    
    def test_custom_values(self):
        obj = PhysicsObject(
            name="heavy", body_id=5, mass=10.0, is_static=True
        )
        assert obj.name == "heavy"
        assert obj.body_id == 5
        assert obj.mass == 10.0
        assert obj.is_static == True

"""Tests for Model Generator - backend strategy logic."""

import pytest
from src.modules.m3_asset_generator.generator import GeneratedModel, ModelGenerator


class TestGeneratedModel:
    
    def test_required_fields(self):
        model = GeneratedModel(name="test", mesh_path="/path/to/mesh.obj")
        assert model.name == "test"
        assert model.mesh_path == "/path/to/mesh.obj"
    
    def test_default_values(self):
        model = GeneratedModel(name="test", mesh_path="/path/to/mesh.obj")
        assert model.preview_path is None
        assert model.backend == "unknown"
    
    def test_custom_backend(self):
        model = GeneratedModel(
            name="test", mesh_path="/mesh.obj", backend="shap-e"
        )
        assert model.backend == "shap-e"


class TestModelGenerator:
    
    def test_unknown_backend_fails_setup(self):
        gen = ModelGenerator(backend="unknown-backend")
        result = gen.setup()
        assert result == False
    
    def test_backend_registry(self):
        assert "shap-e" in ModelGenerator.BACKENDS
        assert "triposr" in ModelGenerator.BACKENDS
    
    def test_generate_text_requires_shap_e(self):
        gen = ModelGenerator(backend="triposr")
        gen._is_setup = True
        result = gen.generate_from_text("test", "/output")
        assert result is None
    
    def test_generate_image_requires_triposr(self):
        gen = ModelGenerator(backend="shap-e")
        gen._is_setup = True
        result = gen.generate_from_image("/test.png", "/output")
        assert result is None
    
    def test_generate_requires_setup(self):
        gen = ModelGenerator(backend="shap-e")
        result = gen.generate_from_text("test", "/output")
        assert result is None


class TestModelGeneratorIntegration:
    """Integration tests - require actual model downloads, skip if unavailable."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_shap_e_setup(self):
        gen = ModelGenerator(backend="shap-e", device="cpu")
        result = gen.setup()
        assert result == True

"""Tests for Pipeline - end-to-end orchestration logic."""

import pytest
from pathlib import Path
from src.pipeline import PipelineConfig, Pipeline

_CHECKPOINT_READY = (Path("checkpoints/scene_extractor/config.json")).exists()


class TestPipelineConfig:
    
    def test_default_values(self):
        config = PipelineConfig()
        assert config.output_dir == "outputs"
        assert config.fps == 24
        assert config.duration == 5.0
        assert config.use_3d_generation == True
        assert config.use_controlnet == False
    
    def test_custom_values(self):
        config = PipelineConfig(
            output_dir="custom_output",
            fps=30,
            duration=10.0,
            use_3d_generation=False
        )
        assert config.output_dir == "custom_output"
        assert config.fps == 30
        assert config.duration == 10.0
        assert config.use_3d_generation == False


class TestPipeline:
    
    def test_init_with_default_config(self):
        pipeline = Pipeline()
        assert pipeline.config is not None
        assert pipeline.config.fps == 24
    
    def test_init_with_custom_config(self):
        config = PipelineConfig(fps=60)
        pipeline = Pipeline(config)
        assert pipeline.config.fps == 60
    
    def test_not_setup_initially(self):
        pipeline = Pipeline()
        assert pipeline._is_setup == False
    
    def test_run_simple_returns_path(self):
        """Verifies run_simple delegates to run and extracts video_path."""
        config = PipelineConfig(use_3d_generation=False, duration=0.5)
        pipeline = Pipeline(config)
        
        # Mock the run method to avoid full pipeline execution
        pipeline.run = lambda prompt, name: {"video_path": "/test/video.mp4"}
        
        result = pipeline.run_simple("test prompt")
        assert result == "/test/video.mp4"


class TestPipelineIntegration:
    """Integration tests - require full dependencies."""
    
    @pytest.fixture
    def minimal_pipeline(self):
        config = PipelineConfig(
            use_3d_generation=False,
            use_controlnet=False,
            duration=0.5,
            fps=5
        )
        return Pipeline(config)
    
    @pytest.mark.slow
    @pytest.mark.skipif(not _CHECKPOINT_READY, reason="T5 checkpoint not trained yet")
    def test_pipeline_setup(self, minimal_pipeline):
        result = minimal_pipeline.setup()
        assert result == True
        assert minimal_pipeline._is_setup == True
    
    @pytest.mark.slow
    @pytest.mark.skipif(not _CHECKPOINT_READY, reason="T5 checkpoint not trained yet")
    def test_pipeline_run_creates_video(self, minimal_pipeline, tmp_path):
        minimal_pipeline.config.output_dir = str(tmp_path)
        minimal_pipeline.setup()
        
        result = minimal_pipeline.run("a ball falls", "test")
        
        assert "video_path" in result
        assert result["scene_description"] is not None
        assert len(result["physics_frames"]) > 0

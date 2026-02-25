"""Tests for KIT-ML Data Loader."""

import pytest
import numpy as np
from src.data import KITMLLoader, KITMLDataset, MotionSample, get_kit_ml_stats


class TestKITMLLoader:
    
    @pytest.fixture
    def loader(self):
        return KITMLLoader("data/KIT-ML")
    
    def test_load_normalization(self, loader):
        mean, std = loader.load_normalization()
        assert mean.shape == (251,)
        assert std.shape == (251,)
    
    def test_load_split(self, loader):
        train_ids = loader.load_split("train")
        val_ids = loader.load_split("val")
        test_ids = loader.load_split("test")
        
        assert len(train_ids) > 0
        assert len(val_ids) > 0
        assert len(test_ids) > 0
    
    def test_load_text(self, loader):
        train_ids = loader.load_split("train")[:5]
        
        for sid in train_ids:
            text = loader.load_text(sid)
            assert isinstance(text, str)
    
    def test_load_motion(self, loader):
        train_ids = loader.load_split("train")[:5]
        
        for sid in train_ids:
            motion = loader.load_motion(sid, normalize=False)
            if motion is not None:
                assert motion.ndim == 2
                assert motion.shape[1] == 251
    
    def test_load_sample(self, loader):
        train_ids = loader.load_split("train")[:5]
        
        for sid in train_ids:
            sample = loader.load_sample(sid)
            if sample:
                assert isinstance(sample, MotionSample)
                assert isinstance(sample.motion, np.ndarray)
                assert sample.duration > 0
    
    def test_load_dataset(self, loader):
        dataset = loader.load_dataset("train", max_samples=10)
        
        assert isinstance(dataset, KITMLDataset)
        assert len(dataset) > 0
        assert dataset.mean is not None
        assert dataset.std is not None
    
    def test_action_distribution(self, loader):
        dataset = loader.load_dataset("train", max_samples=100)
        actions = loader.get_action_distribution(dataset)
        
        assert isinstance(actions, dict)
        assert "walk" in actions


class TestKITMLStats:
    
    def test_get_stats(self):
        stats = get_kit_ml_stats()
        
        assert "train_count" in stats
        assert "val_count" in stats
        assert "motion_dim" in stats
        assert stats["motion_dim"] == 251
        assert stats["fps"] == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

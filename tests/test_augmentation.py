"""Tests for src/data/augmentation — preprocessing + online augmentation."""

from __future__ import annotations

import unittest

import numpy as np

from src.shared.data.augmentation import (
    AugmentationPipeline,
    add_noise,
    detect_tpose,
    quality_filter,
    resample_to_fps,
    speed_perturbation,
    temporal_crop,
)


def _make_smplx_motion(T: int = 100, fps: float = 30.0) -> np.ndarray:
    """Create a synthetic (T, 168) SMPL-X motion sequence."""
    rng = np.random.default_rng(42)
    motion = rng.normal(0, 0.1, (T, 168)).astype(np.float32)
    # Reasonable root translation: gentle walk along +X
    motion[:, 3] = np.linspace(0, 2.0, T)       # X translation
    motion[:, 4] = 0.9 + 0.02 * np.sin(np.linspace(0, 4 * np.pi, T))  # Y (pelvis height)
    motion[:, 5] = np.linspace(0, 0.5, T)       # Z translation
    return motion


class TestResampleToFps(unittest.TestCase):
    def test_60_to_30(self):
        motion = _make_smplx_motion(120, 60.0)
        out = resample_to_fps(motion, 60.0, 30.0)
        self.assertEqual(out.shape[1], 168)
        self.assertEqual(out.shape[0], 60)

    def test_30_to_30_noop(self):
        motion = _make_smplx_motion(100, 30.0)
        out = resample_to_fps(motion, 30.0, 30.0)
        np.testing.assert_array_equal(out, motion)

    def test_20_to_30(self):
        motion = _make_smplx_motion(60, 20.0)
        out = resample_to_fps(motion, 20.0, 30.0)
        # 60 frames at 20fps = 3s → 90 frames at 30fps
        self.assertEqual(out.shape[0], 90)
        self.assertEqual(out.shape[1], 168)


class TestQualityFilter(unittest.TestCase):
    def test_good_motion_passes(self):
        motion = _make_smplx_motion(100)
        self.assertTrue(quality_filter(motion, fps=30.0))

    def test_too_short_fails(self):
        motion = _make_smplx_motion(10)
        self.assertFalse(quality_filter(motion, fps=30.0, min_frames=30))

    def test_exploding_velocity_fails(self):
        motion = _make_smplx_motion(100)
        # Inject unrealistic teleportation
        motion[50, 3] += 100.0  # 100m jump in one frame
        self.assertFalse(quality_filter(motion, fps=30.0, max_root_speed=10.0))


class TestDetectTpose(unittest.TestCase):
    def test_no_tpose(self):
        motion = _make_smplx_motion(100)
        trim_s, trim_e = detect_tpose(motion)
        # Normal motion has high variance — should not detect T-pose
        self.assertEqual(trim_s, 0)
        self.assertEqual(trim_e, 0)

    def test_leading_tpose(self):
        motion = _make_smplx_motion(100)
        motion[:5] = 0.0  # First 5 frames are static
        trim_s, trim_e = detect_tpose(motion, variance_threshold=0.001)
        self.assertGreater(trim_s, 0)

    def test_short_sequence_noop(self):
        motion = _make_smplx_motion(10)
        trim_s, trim_e = detect_tpose(motion)
        self.assertEqual(trim_s, 0)
        self.assertEqual(trim_e, 0)


class TestTemporalCrop(unittest.TestCase):
    def test_shorter_than_max_no_crop(self):
        motion = _make_smplx_motion(50)
        out = temporal_crop(motion, max_length=100)
        np.testing.assert_array_equal(out, motion)

    def test_longer_than_max_cropped(self):
        motion = _make_smplx_motion(200)
        out = temporal_crop(motion, max_length=100)
        self.assertEqual(out.shape[0], 100)
        self.assertEqual(out.shape[1], 168)

    def test_deterministic_with_seed(self):
        motion = _make_smplx_motion(200)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        out1 = temporal_crop(motion, 100, rng1)
        out2 = temporal_crop(motion, 100, rng2)
        np.testing.assert_array_equal(out1, out2)


class TestSpeedPerturbation(unittest.TestCase):
    def test_output_shape(self):
        motion = _make_smplx_motion(100)
        out = speed_perturbation(motion, rng=np.random.default_rng(42))
        self.assertEqual(out.shape[1], 168)
        # Length should vary but stay reasonable (80–120% of original)
        self.assertGreater(out.shape[0], 50)
        self.assertLess(out.shape[0], 150)

    def test_different_seeds_different_output(self):
        motion = _make_smplx_motion(100)
        out1 = speed_perturbation(motion, rng=np.random.default_rng(1))
        out2 = speed_perturbation(motion, rng=np.random.default_rng(2))
        self.assertNotEqual(out1.shape[0], out2.shape[0])


class TestAddNoise(unittest.TestCase):
    def test_shape_preserved(self):
        motion = _make_smplx_motion(100)
        out = add_noise(motion, sigma=0.01)
        self.assertEqual(out.shape, motion.shape)

    def test_translation_unchanged(self):
        motion = _make_smplx_motion(100)
        out = add_noise(motion, sigma=0.01, rng=np.random.default_rng(42))
        # Channels 3:6 (translation) should be unchanged
        np.testing.assert_array_equal(out[:, 3:6], motion[:, 3:6])

    def test_pose_channels_changed(self):
        motion = _make_smplx_motion(100)
        out = add_noise(motion, sigma=0.01, rng=np.random.default_rng(42))
        # Other channels should be perturbed
        self.assertFalse(np.array_equal(out[:, 6:69], motion[:, 6:69]))

    def test_noise_magnitude(self):
        motion = np.zeros((100, 168), dtype=np.float32)
        out = add_noise(motion, sigma=0.01, rng=np.random.default_rng(42))
        # Noise std should be approximately sigma
        measured_std = out[:, 0].std()
        self.assertAlmostEqual(measured_std, 0.01, places=2)


class TestAugmentationPipeline(unittest.TestCase):
    def test_all_enabled(self):
        aug = AugmentationPipeline(seed=42, max_length=80)
        motion = _make_smplx_motion(200)
        out = aug(motion)
        self.assertEqual(out.shape[1], 168)
        self.assertLessEqual(out.shape[0], 80)

    def test_all_disabled(self):
        aug = AugmentationPipeline(
            temporal_crop_enabled=False,
            speed_perturb_enabled=False,
            noise_enabled=False,
            seed=42,
        )
        motion = _make_smplx_motion(100)
        out = aug(motion)
        np.testing.assert_array_equal(out, motion)


if __name__ == "__main__":
    unittest.main()

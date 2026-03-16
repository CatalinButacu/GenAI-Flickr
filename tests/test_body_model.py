"""Tests for the parametric body model and silhouette renderer.

Covers:
  - BodyParams construction, defaults, and property calculations
  - body_params_from_prompt keyword extraction
  - Per-limb width computations (anthropometric proportions)
  - Silhouette rendering (filled body on canvas)
  - SilhouetteSkeletonRenderer API (drop-in for PhysicsSkeletonRenderer)
  - SilhouetteProjector API (drop-in for SkeletonProjector)
  - Pipeline integration (body_params flows through ParsedEntity → PlannedEntity)
  - SMPL body model (when model files are available)
"""

from __future__ import annotations

import unittest

import numpy as np

from src.modules.physics.body_model import (
    BodyParams,
    BodyType,
    _LimbWidths,
    _compute_limb_widths,
    body_params_from_prompt,
    estimate_px_per_m,
    render_body_silhouette,
)
from src.modules.physics.silhouette_renderer import (
    SilhouetteProjector,
    SilhouetteSkeletonRenderer,
)
from src.modules.physics.smpl_body import (
    HAS_SMPL,
    KIT_TO_SMPL,
    SMPL_PARENTS,
    is_smpl_available,
    smpl_betas_from_body_params,
    _rotation_between,
    _rotmat_to_axis_angle,
)


def _make_t_pose_skeleton() -> np.ndarray:
    """Create a synthetic T-pose (21, 3) skeleton in Y-up mm coords.

    Approximate human proportions at 1.72m height.
    """
    joints = np.zeros((21, 3), dtype=np.float64)
    # Spine + head (along Y axis)
    joints[0]  = [0, 0, 0]          # root / pelvis
    joints[1]  = [0, 200, 0]        # spine1
    joints[2]  = [0, 500, 0]        # spine2 / chest
    joints[3]  = [0, 600, 0]        # neck
    joints[4]  = [0, 720, 0]        # head top

    # Left arm (extending -X from shoulder)
    joints[5]  = [-200, 550, 0]     # L shoulder
    joints[6]  = [-450, 550, 0]     # L elbow
    joints[7]  = [-650, 550, 0]     # L wrist

    # Right arm (extending +X from shoulder)
    joints[8]  = [200, 550, 0]      # R shoulder
    joints[9]  = [450, 550, 0]      # R elbow
    joints[10] = [650, 550, 0]      # R wrist

    # Left leg
    joints[11] = [-100, -10, 0]     # L hip
    joints[12] = [-100, -400, 0]    # L knee
    joints[13] = [-100, -800, 0]    # L ankle
    joints[14] = [-100, -820, 50]   # L heel
    joints[15] = [-100, -820, 130]  # L toe

    # Right leg
    joints[16] = [100, -10, 0]      # R hip
    joints[17] = [100, -400, 0]     # R knee
    joints[18] = [100, -800, 0]     # R ankle
    joints[19] = [100, -820, 50]    # R heel
    joints[20] = [100, -820, 130]   # R toe

    return joints


class TestBodyParams(unittest.TestCase):
    """Tests for BodyParams dataclass."""

    def test_defaults(self):
        bp = BodyParams()
        self.assertAlmostEqual(bp.height_m, 1.72)
        self.assertAlmostEqual(bp.mass_kg, 72.0)
        self.assertEqual(bp.gender, "neutral")
        self.assertEqual(bp.age_group, "adult")
        self.assertEqual(bp.body_type, BodyType.AVERAGE)

    def test_bmi(self):
        bp = BodyParams(height_m=1.80, mass_kg=80.0)
        expected = 80.0 / (1.80 ** 2)
        self.assertAlmostEqual(bp.bmi, expected, places=2)

    def test_bulk_factor_average(self):
        bp = BodyParams()
        # bulk_factor = 0.6 + 0.25 * 1.0 + 0.15 * 1.0 = 1.0
        self.assertAlmostEqual(bp.bulk_factor, 1.0)

    def test_bulk_factor_muscular(self):
        bp = BodyParams(muscle_mass=1.5, body_fat=0.5)
        bulk = 0.6 + 0.25 * 1.5 + 0.15 * 0.5
        self.assertAlmostEqual(bp.bulk_factor, bulk)

    def test_apply_body_type_mesomorph(self):
        bp = BodyParams(body_type=BodyType.MESOMORPH, shoulder_width=1.0)
        bp.apply_body_type_defaults()
        self.assertGreaterEqual(bp.muscle_mass, 1.2)
        self.assertGreater(bp.shoulder_width, 1.0)

    def test_apply_body_type_ectomorph(self):
        bp = BodyParams(body_type=BodyType.ECTOMORPH, muscle_mass=1.0)
        bp.apply_body_type_defaults()
        self.assertLessEqual(bp.muscle_mass, 0.7)

    def test_apply_gender_male(self):
        bp = BodyParams(gender="male", shoulder_width=0.9)
        bp.apply_gender_defaults()
        self.assertGreaterEqual(bp.shoulder_width, 1.05)

    def test_apply_gender_female(self):
        bp = BodyParams(gender="female", hip_width=0.9)
        bp.apply_gender_defaults()
        self.assertGreaterEqual(bp.hip_width, 1.05)

    def test_apply_age_child(self):
        bp = BodyParams(age_group="child", head_scale=1.0, height_m=1.20)
        bp.apply_age_defaults()
        self.assertGreaterEqual(bp.head_scale, 1.3)


class TestBodyParamsFromPrompt(unittest.TestCase):
    """Tests for prompt → body params extraction."""

    def test_tall_muscular_man(self):
        bp = body_params_from_prompt("a tall muscular man walks")
        self.assertAlmostEqual(bp.height_m, 1.85)
        self.assertEqual(bp.gender, "male")
        self.assertGreater(bp.muscle_mass, 1.2)
        self.assertEqual(bp.body_type, BodyType.MESOMORPH)

    def test_short_woman(self):
        bp = body_params_from_prompt("a short woman dances")
        self.assertAlmostEqual(bp.height_m, 1.60)
        self.assertEqual(bp.gender, "female")

    def test_chubby_old_person(self):
        bp = body_params_from_prompt("a chubby old man sits")
        self.assertEqual(bp.age_group, "elderly")
        self.assertGreater(bp.body_fat, 1.0)

    def test_default_no_keywords(self):
        bp = body_params_from_prompt("someone moves around")
        self.assertAlmostEqual(bp.height_m, 1.72)
        self.assertEqual(bp.gender, "neutral")

    def test_athletic_boy(self):
        bp = body_params_from_prompt("an athletic boy jumps")
        self.assertEqual(bp.gender, "male")
        self.assertEqual(bp.body_type, BodyType.MESOMORPH)
        self.assertGreater(bp.muscle_mass, 1.0)

    def test_petite_girl(self):
        bp = body_params_from_prompt("a petite girl waves")
        self.assertAlmostEqual(bp.height_m, 1.55)
        self.assertEqual(bp.gender, "female")

    def test_entity_name_used(self):
        bp = body_params_from_prompt("someone walks", entity_name="tall man")
        self.assertAlmostEqual(bp.height_m, 1.85)
        self.assertEqual(bp.gender, "male")


class TestLimbWidths(unittest.TestCase):
    """Tests for anatomical limb width computations."""

    def test_default_widths_positive(self):
        bp = BodyParams()
        widths = _compute_limb_widths(bp, px_per_m=200.0)
        self.assertGreater(widths.head_rx, 0)
        self.assertGreater(widths.head_ry, 0)
        self.assertGreater(widths.neck, 0)
        self.assertGreater(widths.shoulder, 0)
        self.assertGreater(widths.upper_arm, 0)
        self.assertGreater(widths.upper_leg, 0)

    def test_muscular_wider_arms(self):
        lean = _compute_limb_widths(BodyParams(muscle_mass=0.5), px_per_m=200.0)
        buff = _compute_limb_widths(BodyParams(muscle_mass=1.5), px_per_m=200.0)
        self.assertGreater(buff.upper_arm, lean.upper_arm)
        self.assertGreater(buff.forearm, lean.forearm)

    def test_taller_larger_head(self):
        short_w = _compute_limb_widths(BodyParams(height_m=1.50), px_per_m=200.0)
        tall_w = _compute_limb_widths(BodyParams(height_m=1.90), px_per_m=200.0)
        self.assertGreater(tall_w.head_rx, short_w.head_rx)

    def test_px_per_m_scaling(self):
        bp = BodyParams()
        w100 = _compute_limb_widths(bp, px_per_m=100.0)
        w300 = _compute_limb_widths(bp, px_per_m=300.0)
        self.assertAlmostEqual(w300.shoulder / w100.shoulder, 3.0, places=1)


class TestRenderBodySilhouette(unittest.TestCase):
    """Tests for the silhouette rendering function."""

    def setUp(self):
        self.skeleton = _make_t_pose_skeleton()

    def test_output_shape(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        bp = BodyParams()
        uvd = np.column_stack([
            self.skeleton[:, 0] * 0.3 + 320,
            -self.skeleton[:, 1] * 0.3 + 240,
            np.zeros(21),
        ])
        result = render_body_silhouette(uvd, bp, canvas, px_per_m=200.0)
        self.assertEqual(result.shape, (480, 640, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_silhouette_not_empty(self):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        bp = BodyParams()
        uvd = np.column_stack([
            self.skeleton[:, 0] * 0.3 + 320,
            -self.skeleton[:, 1] * 0.3 + 240,
            np.zeros(21),
        ])
        result = render_body_silhouette(uvd, bp, canvas, px_per_m=200.0)
        # At least some pixels should be non-zero (body was drawn)
        non_zero = np.count_nonzero(result)
        self.assertGreater(non_zero, 100)

    def test_different_body_params_different_output(self):
        bp_slim = BodyParams(muscle_mass=0.5, body_fat=0.4)
        bp_buff = BodyParams(muscle_mass=1.5, body_fat=0.7)
        uvd = np.column_stack([
            self.skeleton[:, 0] * 0.3 + 320,
            -self.skeleton[:, 1] * 0.3 + 240,
            np.zeros(21),
        ])
        canvas1 = np.zeros((480, 640, 3), dtype=np.uint8)
        canvas2 = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = render_body_silhouette(uvd, bp_slim, canvas1, px_per_m=200.0)
        r2 = render_body_silhouette(uvd, bp_buff, canvas2, px_per_m=200.0)
        # The two silhouettes should be different (buff is wider)
        self.assertFalse(np.array_equal(r1, r2))


class TestEstimatePxPerM(unittest.TestCase):
    """Tests for the pixel-per-metre scale estimation."""

    def test_basic_estimation(self):
        uvd = np.zeros((21, 3))
        uvd[:, 1] = np.linspace(100, 500, 21)  # 400px span
        result = estimate_px_per_m(uvd, body_height_m=1.72)
        expected = 400.0 / 1.72
        self.assertAlmostEqual(result, expected, places=1)

    def test_very_small_skeleton_fallback(self):
        uvd = np.zeros((21, 3))  # all at origin
        result = estimate_px_per_m(uvd, body_height_m=1.72)
        self.assertEqual(result, 200.0)  # fallback


class TestSilhouetteSkeletonRenderer(unittest.TestCase):
    """Tests for the full silhouette renderer."""

    def test_render_frame_returns_image(self):
        renderer = SilhouetteSkeletonRenderer(img_w=640, img_h=480)
        skeleton = _make_t_pose_skeleton()
        frame = renderer.render_frame(skeleton)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_render_frame_with_body_params(self):
        bp = BodyParams(height_m=1.85, muscle_mass=1.3, gender="male")
        renderer = SilhouetteSkeletonRenderer(
            img_w=640, img_h=480, body_params=bp,
        )
        skeleton = _make_t_pose_skeleton()
        frame = renderer.render_frame(skeleton)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_render_frame_per_frame_override(self):
        renderer = SilhouetteSkeletonRenderer(img_w=640, img_h=480)
        skeleton = _make_t_pose_skeleton()
        bp_override = BodyParams(height_m=1.50, body_fat=1.5)
        frame = renderer.render_frame(skeleton, body_params=bp_override)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_update_camera(self):
        renderer = SilhouetteSkeletonRenderer(img_w=640, img_h=480)
        # Should not raise
        renderer.update_camera(45.0, -25.0, 4.0, [0.0, 900.0, 0.0])


class TestSilhouetteProjector(unittest.TestCase):
    """Tests for the ControlNet conditioning silhouette projector."""

    def test_project_shape(self):
        projector = SilhouetteProjector(img_w=512, img_h=512)
        skeleton = _make_t_pose_skeleton()
        uvd = projector.project(skeleton)
        self.assertEqual(uvd.shape, (21, 3))

    def test_render_returns_image(self):
        projector = SilhouetteProjector(img_w=512, img_h=512)
        skeleton = _make_t_pose_skeleton()
        img = projector.render(skeleton)
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertEqual(img.dtype, np.uint8)

    def test_render_with_body_params(self):
        bp = BodyParams(height_m=1.60, gender="female", body_fat=1.3)
        projector = SilhouetteProjector(img_w=512, img_h=512, body_params=bp)
        skeleton = _make_t_pose_skeleton()
        img = projector.render(skeleton)
        self.assertEqual(img.shape, (512, 512, 3))
        # Image should have non-zero pixels (silhouette drawn)
        self.assertGreater(np.count_nonzero(img), 100)


class TestPipelineIntegration(unittest.TestCase):
    """Tests for body_params flowing through the M1/M2 pipeline."""

    def test_parsed_entity_has_body_params(self):
        from src.modules.understanding.prompt_parser import PromptParser
        parser = PromptParser()
        scene = parser.parse("a tall muscular man walks forward")
        actors = [e for e in scene.entities if e.is_actor]
        self.assertGreater(len(actors), 0, "Expected at least one actor")
        actor = actors[0]
        self.assertIsNotNone(actor.body_params, "Actor should have body_params")
        assert actor.body_params is not None  # type narrowing
        self.assertAlmostEqual(actor.body_params.height_m, 1.85)
        self.assertEqual(actor.body_params.gender, "male")

    def test_planned_entity_propagates_body_params(self):
        from src.modules.understanding.prompt_parser import PromptParser
        from src.modules.planner import ScenePlanner
        parser = PromptParser()
        planner = ScenePlanner()
        parsed = parser.parse("a short woman dances")
        planned = planner.plan(parsed)
        actors = [e for e in planned.entities if e.is_actor]
        self.assertGreater(len(actors), 0)
        self.assertIsNotNone(actors[0].body_params)
        assert actors[0].body_params is not None  # type narrowing
        self.assertAlmostEqual(actors[0].body_params.height_m, 1.60)

    def test_non_actor_no_body_params(self):
        from src.modules.understanding.prompt_parser import PromptParser
        parser = PromptParser()
        scene = parser.parse("a red ball bounces")
        objects = [e for e in scene.entities if not e.is_actor]
        for obj in objects:
            self.assertIsNone(obj.body_params)


# ── SMPL module tests (always run — do not require model files) ──────────

class TestSMPLRotationHelpers(unittest.TestCase):
    """Tests for SMPL rotation math utilities."""

    def test_rotation_identity(self):
        """Same direction → identity rotation."""
        v = np.array([0.0, 1.0, 0.0])
        R = _rotation_between(v, v)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-6)

    def test_rotation_90_degrees(self):
        """Rotate X-axis to Y-axis (90°)."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        R = _rotation_between(v1, v2)
        result = R @ v1
        np.testing.assert_allclose(result, v2, atol=1e-6)

    def test_rotation_180_degrees(self):
        """Rotate to opposite direction (180°)."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        R = _rotation_between(v1, v2)
        result = R @ v1
        np.testing.assert_allclose(result, v2, atol=1e-6)

    def test_rotation_arbitrary(self):
        """Arbitrary direction rotation."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([-2.0, 1.0, 0.5])
        R = _rotation_between(v1, v2)
        result = R @ (v1 / np.linalg.norm(v1))
        expected = v2 / np.linalg.norm(v2)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_rotation_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R^T R = I)."""
        v1 = np.array([1.0, 0.5, -0.3])
        v2 = np.array([-0.7, 1.2, 0.8])
        R = _rotation_between(v1, v2)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)

    def test_axis_angle_identity(self):
        """Identity rotation → zero axis-angle."""
        aa = _rotmat_to_axis_angle(np.eye(3))
        np.testing.assert_allclose(aa, np.zeros(3), atol=1e-6)

    def test_axis_angle_roundtrip(self):
        """Axis-angle → rotation → axis-angle roundtrip."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        R = _rotation_between(v1, v2)
        aa = _rotmat_to_axis_angle(R)
        angle = float(np.linalg.norm(aa))
        self.assertAlmostEqual(angle, np.pi / 2, places=4)


class TestSMPLJointMapping(unittest.TestCase):
    """Tests for the 21-joint ↔ SMPL joint correspondence."""

    def test_kit_to_smpl_has_19_entries(self):
        """19 of 21 KIT joints have direct SMPL correspondences."""
        self.assertEqual(len(KIT_TO_SMPL), 19)

    def test_kit_to_smpl_covers_major_joints(self):
        """All major body joints are mapped."""
        # Root, spine, neck, head
        for kit_idx in [0, 1, 2, 3, 4]:
            self.assertIn(kit_idx, KIT_TO_SMPL)
        # Arms
        for kit_idx in [5, 6, 7, 8, 9, 10]:
            self.assertIn(kit_idx, KIT_TO_SMPL)
        # Legs (hips, knees, ankles, toes)
        for kit_idx in [11, 12, 13, 15, 16, 17, 18, 20]:
            self.assertIn(kit_idx, KIT_TO_SMPL)

    def test_heels_not_mapped(self):
        """Heel joints (14, 19) have no direct SMPL correspondence."""
        self.assertNotIn(14, KIT_TO_SMPL)
        self.assertNotIn(19, KIT_TO_SMPL)

    def test_smpl_parents_length(self):
        """SMPL has 24 joints."""
        self.assertEqual(len(SMPL_PARENTS), 24)

    def test_smpl_root_has_no_parent(self):
        self.assertEqual(SMPL_PARENTS[0], -1)


class TestSMPLBetasMapping(unittest.TestCase):
    """Tests for BodyParams → SMPL betas conversion."""

    def test_default_params_near_zero_betas(self):
        """Default BodyParams → near-zero betas (average body)."""
        bp = BodyParams()
        betas = smpl_betas_from_body_params(bp)
        self.assertEqual(betas.shape, (10,))
        # Most betas should be close to zero for average body
        self.assertAlmostEqual(float(betas[0]), 0.0, places=1)  # average height
        self.assertAlmostEqual(float(betas[1]), 0.0, places=1)  # average bulk

    def test_tall_person_positive_beta0(self):
        """Tall body → positive β₀ (height component)."""
        bp = BodyParams(height_m=1.92)
        betas = smpl_betas_from_body_params(bp)
        self.assertGreater(betas[0], 0.5)

    def test_short_person_negative_beta0(self):
        """Short body → negative β₀."""
        bp = BodyParams(height_m=1.52)
        betas = smpl_betas_from_body_params(bp)
        self.assertLess(betas[0], -0.5)

    def test_muscular_wider_betas(self):
        """Muscular build → positive bulk-related betas."""
        bp = BodyParams(muscle_mass=1.5, body_fat=0.7)
        betas = smpl_betas_from_body_params(bp)
        # Bulk factor > 1.0 → positive β₁
        self.assertGreater(betas[1], 0.0)

    def test_male_female_different_betas(self):
        """Male and female bodies produce different shape betas."""
        bp_m = BodyParams(gender="male")
        bp_m.apply_gender_defaults()
        bp_f = BodyParams(gender="female")
        bp_f.apply_gender_defaults()
        betas_m = smpl_betas_from_body_params(bp_m)
        betas_f = smpl_betas_from_body_params(bp_f)
        # Gender betas (8, 9) should differ
        self.assertNotAlmostEqual(float(betas_m[8]), float(betas_f[8]), places=1)

    def test_body_params_to_smpl_betas_method(self):
        """BodyParams.to_smpl_betas() convenience method."""
        bp = BodyParams(height_m=1.85, muscle_mass=1.3)
        betas = bp.to_smpl_betas()
        self.assertEqual(betas.shape, (10,))
        self.assertGreater(betas[0], 0.0)  # tall → positive height beta


class TestSMPLAvailability(unittest.TestCase):
    """Tests for the SMPL availability check."""

    def test_has_smpl_is_bool(self):
        self.assertIsInstance(HAS_SMPL, bool)

    def test_is_smpl_available_returns_bool(self):
        result = is_smpl_available()
        self.assertIsInstance(result, bool)

    def test_is_smpl_available_nonexistent_gender(self):
        result = is_smpl_available(gender="nonexistent_gender")
        self.assertFalse(result)


@unittest.skipUnless(is_smpl_available(), "SMPL model files not installed")
class TestSMPLBodyModel(unittest.TestCase):
    """Tests for SMPLBody — only run when SMPL model files are available."""

    @classmethod
    def setUpClass(cls):
        from src.modules.physics.smpl_body import SMPLBody
        cls.body = SMPLBody.get_or_create(gender="neutral")

    def test_rest_vertices_shape(self):
        self.assertEqual(self.body.rest_vertices.shape, (6890, 3))

    def test_rest_joints_shape(self):
        self.assertEqual(self.body.rest_joints.shape, (24, 3))

    def test_faces_shape(self):
        self.assertEqual(self.body.faces.shape[1], 3)
        self.assertGreater(self.body.n_faces, 10000)

    def test_get_shaped_mesh_default(self):
        verts, faces = self.body.get_shaped_mesh()
        self.assertEqual(verts.shape, (6890, 3))

    def test_get_shaped_mesh_custom_betas(self):
        betas = np.zeros(10, dtype=np.float32)
        betas[0] = 2.0  # taller
        verts, faces = self.body.get_shaped_mesh(betas=betas)
        # Taller body should have greater Y span
        default_height = self.body.rest_vertices[:, 1].max() - self.body.rest_vertices[:, 1].min()
        custom_height = verts[:, 1].max() - verts[:, 1].min()
        self.assertGreater(custom_height, default_height * 0.95)

    def test_get_posed_mesh(self):
        skeleton = _make_t_pose_skeleton()
        verts, faces = self.body.get_posed_mesh(skeleton)
        self.assertEqual(verts.shape, (6890, 3))
        # Vertices should be in mm (SMPL scale)
        self.assertGreater(verts[:, 1].max() - verts[:, 1].min(), 100)  # at least 100mm

    def test_caching(self):
        from src.modules.physics.smpl_body import SMPLBody
        body1 = SMPLBody.get_or_create(gender="neutral")
        body2 = SMPLBody.get_or_create(gender="neutral")
        self.assertIs(body1, body2)


if __name__ == "__main__":
    unittest.main()

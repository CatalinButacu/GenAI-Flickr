"""End-to-end integration test: Text → Pipeline → Video using real AMASS data.

Tests the full pipeline with SMPL-X motion data from AMASS, ConceptNet KB
enrichment, physics simulation, and video export.
"""

import logging
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def test_data_loaders():
    """Test that AMASS and ARCTIC data loaders work."""
    log.info("=" * 60)
    log.info("TEST 1: Data Loaders")
    log.info("=" * 60)

    from src.shared.data.smplx_loader import AMASSLoader, ARCTICLoader

    amass = AMASSLoader()
    files = amass.discover_files()
    log.info("AMASS: %d files found", len(files))
    assert len(files) > 0, "No AMASS files found"

    sample = amass.load_file(files[0])
    assert sample is not None, "Failed to load AMASS sample"
    assert sample.motion.shape[1] == 168, f"Expected 168 dims, got {sample.motion.shape[1]}"
    log.info("AMASS sample: %s, shape=%s, fps=%.0f", sample.sample_id, sample.motion.shape, sample.fps)

    arctic = ARCTICLoader()
    seqs = arctic.discover_sequences()
    log.info("ARCTIC: %d sequences found", len(seqs))
    if seqs:
        s2 = arctic.load_sequence(seqs[0][0], seqs[0][1])
        assert s2 is not None
        assert s2.motion.shape[1] == 168
        log.info("ARCTIC sample: %s, shape=%s, obj=%s",
                 s2.sample_id, s2.motion.shape,
                 s2.object_motion.shape if s2.object_motion is not None else None)

    log.info("TEST 1 PASSED\n")


def test_smplx_body():
    """Test SMPL-X forward kinematics."""
    log.info("=" * 60)
    log.info("TEST 2: SMPL-X Body Model")
    log.info("=" * 60)

    from src.modules.physics.smplx_body import SMPLXBody
    from src.shared.data.smplx_loader import AMASSLoader

    body = SMPLXBody.get_or_create("neutral")
    log.info("SMPL-X body: %d vertices, %d joints, %d faces",
             body.n_vertices, body.n_joints, body.n_faces)

    loader = AMASSLoader()
    sample = loader.load_file(loader.discover_files()[0])
    verts, joints = body.forward_from_pose_vector(sample.motion[0], sample.betas)

    assert verts.shape == (10475, 3), f"Expected (10475, 3), got {verts.shape}"
    assert joints.shape == (55, 3), f"Expected (55, 3), got {joints.shape}"
    log.info("Forward pass: verts=%s, joints=%s", verts.shape, joints.shape)
    log.info("TEST 2 PASSED\n")


def test_physics_extraction():
    """Test physics state extraction from SMPL-X data."""
    log.info("=" * 60)
    log.info("TEST 3: Physics State Extraction")
    log.info("=" * 60)

    from src.shared.data.physics_dataset import extract_physics_state
    from src.shared.data.smplx_loader import AMASSLoader

    loader = AMASSLoader()
    sample = loader.load_file(loader.discover_files()[0])
    phys = extract_physics_state(sample.motion, fps=int(sample.fps))

    assert phys.shape == (sample.motion.shape[0], 64)
    nz = int((abs(phys).sum(axis=0) > 1e-6).sum())
    log.info("Physics state: shape=%s, non-zero channels=%d/64", phys.shape, nz)
    assert nz >= 20, f"Expected >=20 non-zero channels, got {nz}"
    log.info("TEST 3 PASSED\n")


def test_motion_retargeting():
    """Test SMPL-X → PyBullet retargeting."""
    log.info("=" * 60)
    log.info("TEST 4: Motion Retargeting (55 joints)")
    log.info("=" * 60)

    from src.modules.motion.amass_retriever import AMASSSampleRetriever
    from src.modules.physics.motion_retarget import retarget_sequence

    retriever = AMASSSampleRetriever(max_samples=3)
    clip = retriever.retrieve("walking", max_frames=30)

    assert clip is not None
    assert clip.raw_joints is not None
    assert clip.raw_joints.shape[1] == 55

    angles, transforms = retarget_sequence(clip.raw_joints)
    assert len(angles) == 30
    assert len(transforms) == 30
    assert "left_knee" in angles[0]
    log.info("Retargeting OK: %d frames with %d joint angles each",
             len(angles), len(angles[0]))
    log.info("TEST 4 PASSED\n")


def test_conceptnet_kb():
    """Test ConceptNet knowledge base retrieval."""
    log.info("=" * 60)
    log.info("TEST 5: ConceptNet KB")
    log.info("=" * 60)

    from src.modules.understanding.retriever import KnowledgeRetriever

    kb = KnowledgeRetriever()
    ok = kb.setup()
    assert ok, "KB setup failed"
    assert kb.entry_count > 10000, f"Expected >10K entries, got {kb.entry_count}"

    results = kb.retrieve("wooden table", top_k=3)
    assert len(results) > 0
    log.info("KB: %d entries, query 'wooden table' → %s",
             kb.entry_count, [r.name for r in results])
    log.info("TEST 5 PASSED\n")


def test_pipeline_e2e():
    """End-to-end pipeline test with real AMASS data."""
    log.info("=" * 60)
    log.info("TEST 6: Full Pipeline (E2E)")
    log.info("=" * 60)

    from src.pipeline import Pipeline, PipelineConfig

    config = PipelineConfig(
        use_t5_parser=False,           # use rules parser for speed
        use_asset_generation=False,    # skip Shap-E
        use_motion_generation=True,    # use AMASS data retrieval
        use_physics_ssm=True,          # use SSM for temporal refinement
        use_diffusion=False,           # skip ControlNet/AnimateDiff
        use_render_engine=True,
        use_silhouette=True,
        validate_motion=False,         # skip biomechanical validation
        fps=30,                        # 30fps output
        fixed_camera=True,             # static viewpoint — no orbit/zoom
        duration=3.0,                  # enough to see full movement
        device="cpu",
    )

    pipeline = Pipeline(config)
    log.info("Running pipeline with prompt: 'A person walks forward near a table'")
    result = pipeline.run("A person walks forward near a table",
                          output_name="test_e2e_smplx")

    assert "error" not in result, f"Pipeline error: {result.get('error')}"
    assert result.get("video_path"), "No video produced"
    log.info("Pipeline output: %s", result["video_path"])

    frame_count = len(result.get("physics_frames", []))
    video_duration = frame_count / max(config.fps, 1)
    file_size = os.path.getsize(result["video_path"]) if os.path.exists(result["video_path"]) else 0

    log.info("  runtime_elapsed: %.2fs", result.get("elapsed_seconds", 0))
    log.info("  video_duration: %.2fs", video_duration)
    log.info("  frames: %d", frame_count)
    log.info("  file_size_bytes: %d", file_size)

    if result.get("motion_clips"):
        for actor, clip in result["motion_clips"].items():
            if clip:
                log.info("  motion %s: frames=%s, source=%s",
                         actor, clip.frames.shape, clip.source)

    log.info("TEST 6 PASSED\n")


def main():
    tests = [
        ("Data Loaders", test_data_loaders),
        ("SMPL-X Body", test_smplx_body),
        ("Physics Extraction", test_physics_extraction),
        ("Motion Retargeting", test_motion_retargeting),
        ("ConceptNet KB", test_conceptnet_kb),
        ("Pipeline E2E", test_pipeline_e2e),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            log.error("TEST FAILED: %s — %s", name, e, exc_info=True)
            failed += 1

    log.info("=" * 60)
    log.info("RESULTS: %d passed, %d failed out of %d tests",
             passed, failed, len(tests))
    log.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

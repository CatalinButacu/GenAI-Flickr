"""
Pipeline Component Test Script
==============================
Tests all modules and creates a summary.
"""

import sys

def test_all():
    results = []
    
    print("=" * 60)
    print("PIPELINE COMPONENT TEST SUMMARY")
    print("=" * 60)
    print()
    
    # Test 1: Shared Vocabulary
    print("[1] Shared Vocabulary...")
    try:
        from src.shared.vocabulary import ACTIONS, OBJECTS, get_action_by_keyword
        get_action_by_keyword("walk")
        results.append(("Shared Vocabulary", True, f"{len(ACTIONS)} actions, {len(OBJECTS)} objects"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Shared Vocabulary", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 2: Prompt Parser
    print("[2] Prompt Parser (M1)...")
    try:
        from src.modules.m1_scene_understanding.prompt_parser import PromptParser
        parser = PromptParser(mode="rules")
        parser.setup()
        parsed = parser.parse("A person walks to a ball")
        results.append(("Prompt Parser", True, f"{len(parsed.entities)} entities, {len(parsed.actions)} actions"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Prompt Parser", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 3: Scene Planner
    print("[3] Scene Planner (M2)...")
    try:
        from src.modules.m2_scene_planner import ScenePlanner
        planner = ScenePlanner()
        planned = planner.plan(parsed)
        results.append(("Scene Planner", True, f"{len(planned.entities)} positioned"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Scene Planner", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 4: Motion Generator
    print("[4] Motion Generator (M4)...")
    try:
        from src.modules.m4_motion_generator import MotionGenerator
        gen = MotionGenerator()
        gen.setup()
        clip = gen.generate("walk", duration=1.0)
        results.append(("Motion Generator", True, f"{len(clip.frames)} frames"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Motion Generator", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 5: SSM Motion Generator
    print("[5] SSM Motion Generator...")
    try:
        from src.modules.m4_motion_generator import SSMMotionGenerator
        gen = SSMMotionGenerator(backend="ssm")
        gen.setup()
        clip = gen.generate("walk", duration=1.0)
        results.append(("SSM Motion Generator", True, f"{len(clip.frames)} SSM frames"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("SSM Motion Generator", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 6: Physics Engine
    print("[6] Physics Engine (M5)...")
    try:
        from src.modules.m5_physics_engine import Scene
        scene = Scene()
        scene.setup()
        scene.add_ground()
        scene.add_primitive("test_ball", "sphere", 0.1, 1.0, [0, 0, 1])
        scene.cleanup()
        results.append(("Physics Engine", True, "Scene created"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Physics Engine", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 7: Humanoid Body
    print("[7] Humanoid Body Loader...")
    try:
        from src.modules.m5_physics_engine.humanoid import HumanoidBody
        results.append(("Humanoid Body", True, "Import OK"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Humanoid Body", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 8: SSM Core
    print("[8] SSM Module...")
    try:
        from src.ssm import get_ssm_info
        info = get_ssm_info()
        layers = len(info["layers"])
        results.append(("SSM Module", True, f"{layers} layers, torch={info['torch_available']}"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("SSM Module", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 9: Pipeline Setup
    print("[9] Pipeline v2 Setup...")
    try:
        from src.pipeline_v2 import Pipeline, PipelineConfig
        config = PipelineConfig(
            use_asset_generation=False,
            use_motion_generation=True,
            use_ai_enhancement=False,
            duration=1.0
        )
        pipeline = Pipeline(config)
        pipeline.setup()
        results.append(("Pipeline Setup", True, "All modules ready"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Pipeline Setup", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Test 10: Full Pipeline Run
    print("[10] Full Pipeline Run...")
    try:
        result = pipeline.run("A red ball falls on blue cube", output_name="test")
        n_frames = len(result["physics_frames"])
        results.append(("Full Pipeline", True, f"{n_frames} frames, video saved"))
        print("    ✓ PASS")
    except Exception as e:
        results.append(("Full Pipeline", False, str(e)))
        print(f"    ✗ FAIL - {e}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    working = [r for r in results if r[1]]
    not_working = [r for r in results if not r[1]]
    
    print(f"\n✓ WORKING ({len(working)}/{len(results)}):")
    for name, _, detail in working:
        print(f"    • {name}: {detail}")
    
    if not_working:
        print(f"\n✗ NOT WORKING ({len(not_working)}/{len(results)}):")
        for name, _, detail in not_working:
            print(f"    • {name}: {detail}")
    
    print()
    print("=" * 60)
    
    return len(not_working) == 0

if __name__ == "__main__":
    success = test_all()
    sys.exit(0 if success else 1)

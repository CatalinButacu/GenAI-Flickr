"""
M5 Physics Engine - Benchmark Tests
=====================================
30 rigorous tests for physics simulation.

Run: py tests/benchmark_m5.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class M5Benchmark:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.scene = None
        self.simulator = None

    def test(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name} {details}")

    def _test_imports(self) -> bool:
        print("\n[1-5] IMPORTS")
        print("-" * 50)

        try:
            from src.modules.m5_physics_engine import Scene, Simulator, CameraConfig, CinematicCamera  # noqa: F401
            self.test("1. Core imports", True)
        except ImportError as e:
            self.test("1. Core imports", False, str(e)[:30])
            return False

        try:
            from src.modules.m5_physics_engine.scene import ShapeFactory, PhysicsObject  # noqa: F401
            self.test("2. Scene classes", True)
        except ImportError as e:
            self.test("2. Scene classes", False, str(e)[:30])

        try:
            from src.modules.m5_physics_engine.simulator import FrameData, EasingFunctions  # noqa: F401
            self.test("3. Simulator classes", True)
        except ImportError as e:
            self.test("3. Simulator classes", False, str(e)[:30])

        try:
            import pybullet  # noqa: F401
            self.test("4. PyBullet installed", True)
        except ImportError:
            self.test("4. PyBullet installed", False)
            return False

        try:
            import pybullet_data  # noqa: F401
            self.test("5. pybullet_data installed", True)
        except ImportError:
            self.test("5. pybullet_data installed", False)

        return True

    def _test_scene_setup(self) -> bool:
        print("\n[6-10] SCENE SETUP")
        print("-" * 50)

        from src.modules.m5_physics_engine import Scene

        try:
            self.scene = Scene(gravity=-9.81)
            self.test("6. Scene init", self.scene is not None)
        except Exception as e:
            self.test("6. Scene init", False, str(e)[:30])
            return False

        try:
            setup_ok = self.scene.setup(use_gui=False)
            self.test("7. Scene setup (headless)", setup_ok)
        except Exception as e:
            self.test("7. Scene setup", False, str(e)[:30])
            return False

        try:
            ground_id = self.scene.add_ground()
            self.test("8. Add ground", ground_id is not None)
        except Exception as e:
            self.test("8. Add ground", False, str(e)[:30])

        try:
            self.test("9. Gravity set", self.scene.gravity == -9.81)
        except Exception as e:
            self.test("9. Gravity set", False, str(e)[:30])

        try:
            self.test("10. Scene is setup flag", self.scene._is_setup)
        except Exception as e:
            self.test("10. Setup flag", False, str(e)[:30])

        return True

    def _test_primitives(self) -> None:
        print("\n[11-16] PRIMITIVES")
        print("-" * 50)

        primitives = [
            ("11. Add box",      "test_box",      "box",      [0.1, 0.1, 0.1], [0, 0, 1],    [1, 0, 0, 1], False),
            ("12. Add sphere",   "test_sphere",   "sphere",   [0.1],            [0.5, 0, 1],  [0, 1, 0, 1], False),
            ("13. Add cylinder", "test_cylinder", "cylinder", [0.05, 0.2],      [1, 0, 1],    [0, 0, 1, 1], False),
            ("14. Static object","static_box",    "box",      [0.5, 0.5, 0.1],  [0, 0, 0.1],  [1, 1, 0, 1], True),
        ]
        for label, name, shape, size, pos, color, is_static in primitives:
            try:
                obj = self.scene.add_primitive(name, shape=shape, size=size, position=pos, color=color, is_static=is_static)
                ok = obj is not None and (obj.is_static if is_static else obj.body_id >= 0)
                self.test(label, ok)
            except Exception as e:
                self.test(label, False, str(e)[:30])

        try:
            obj = self.scene.get_object("test_box")
            self.test("15. Get object", obj is not None and obj.name == "test_box")
        except Exception as e:
            self.test("15. Get object", False, str(e)[:30])

        try:
            pos, orn = self.scene.get_object_state("test_box")
            self.test("16. Get state", len(pos) == 3 and len(orn) == 4)
        except Exception as e:
            self.test("16. Get state", False, str(e)[:30])

    def _test_simulation(self) -> bool:
        print("\n[17-22] PHYSICS SIMULATION")
        print("-" * 50)

        from src.modules.m5_physics_engine import Simulator, CameraConfig

        try:
            camera = CameraConfig(width=320, height=240)
            self.simulator = Simulator(self.scene, camera)
            self.test("17. Simulator init", self.simulator is not None)
        except Exception as e:
            self.test("17. Simulator init", False, str(e)[:30])
            return False

        try:
            self.simulator.step()
            self.test("18. Step simulation", True)
        except Exception as e:
            self.test("18. Step simulation", False, str(e)[:30])

        frame = None
        try:
            frame = self.simulator.render()
            self.test("19. Render frame", frame is not None and frame.rgb is not None)
        except Exception as e:
            self.test("19. Render frame", False, str(e)[:30])

        try:
            self.test("20. RGB shape", frame.rgb.shape[2] == 3, f"shape={frame.rgb.shape}")
        except Exception as e:
            self.test("20. RGB shape", False, str(e)[:30])

        try:
            self.test("21. Depth available", frame.depth is not None)
        except Exception as e:
            self.test("21. Depth available", False, str(e)[:30])

        try:
            self.scene.apply_force("test_sphere", [0, 0, 10])
            self.test("22. Apply force", True)
        except Exception as e:
            self.test("22. Apply force", False, str(e)[:30])

        return True

    def _test_cinematic_camera(self) -> None:
        print("\n[23-26] CINEMATIC CAMERA")
        print("-" * 50)

        from src.modules.m5_physics_engine import CinematicCamera

        try:
            cine = CinematicCamera(distance=2.0, yaw=0, pitch=-20)
            self.test("23. CinematicCamera init", cine is not None)
        except Exception as e:
            self.test("23. CinematicCamera init", False, str(e)[:30])
            return

        try:
            cine.add_orbit(start_yaw=0, end_yaw=90, duration=2.0)
            self.test("24. Add orbit", len(cine.effects) > 0)
        except Exception as e:
            self.test("24. Add orbit", False, str(e)[:30])

        try:
            cine.add_zoom(start_dist=2.0, end_dist=1.5, duration=1.0)
            self.test("25. Add zoom", len(cine.effects) > 1)
        except Exception as e:
            self.test("25. Add zoom", False, str(e)[:30])

        try:
            cine.update(1.0)
            params = cine.get_camera_params()
            self.test("26. Camera params", len(params) == 4 and isinstance(params[0], float))
        except Exception as e:
            self.test("26. Camera params", False, str(e)[:30])

    def _test_video(self) -> None:
        print("\n[27-28] VIDEO GENERATION")
        print("-" * 50)

        frames = []
        try:
            frames = self.simulator.run(duration=0.5, fps=10)
            self.test("27. Run simulation", len(frames) > 3, f"frames={len(frames)}")
        except Exception as e:
            self.test("27. Run simulation", False, str(e)[:30])

        try:
            output_dir = "outputs/physics_test"
            os.makedirs(output_dir, exist_ok=True)
            video_path = f"{output_dir}/test_video.mp4"
            self.simulator.create_video(frames, video_path, fps=10)
            self.test("28. Create video", os.path.exists(video_path))
        except Exception as e:
            self.test("28. Create video", False, str(e)[:30])

    def _test_physics_accuracy(self) -> None:
        print("\n[29-30] PHYSICS ACCURACY")
        print("-" * 50)

        from src.modules.m5_physics_engine import Scene, Simulator, CameraConfig

        try:
            drop_scene = Scene(gravity=-9.81)
            drop_scene.setup()
            drop_scene.add_ground()
            drop_scene.add_primitive("drop_ball", shape="sphere", size=[0.1], position=[0, 0, 2], mass=1.0)
            drop_sim = Simulator(drop_scene, CameraConfig())
            pos0, _ = drop_scene.get_object_state("drop_ball")
            for _ in range(120):
                drop_sim.step()
            pos1, _ = drop_scene.get_object_state("drop_ball")
            self.test("29. Gravity works (ball falls)", pos1[2] < pos0[2], f"z: {pos0[2]:.2f} -> {pos1[2]:.2f}")
            drop_scene.cleanup()
        except Exception as e:
            self.test("29. Gravity works", False, str(e)[:30])

        try:
            coll_scene = Scene(gravity=-9.81)
            coll_scene.setup()
            coll_scene.add_ground()
            coll_scene.add_primitive("floor_ball", shape="sphere", size=[0.1], position=[0, 0, 0.2], mass=1.0)
            coll_sim = Simulator(coll_scene, CameraConfig())
            for _ in range(240):
                coll_sim.step()
            pos, _ = coll_scene.get_object_state("floor_ball")
            self.test("30. Collision (ball on ground)", 0.05 < pos[2] < 0.5, f"z={pos[2]:.3f}")
            coll_scene.cleanup()
        except Exception as e:
            self.test("30. Collision", False, str(e)[:30])

    def run_all(self):
        print("=" * 70)
        print("M5 PHYSICS ENGINE - BENCHMARK SUITE")
        print("=" * 70)

        if not self._test_imports():
            return False
        if not self._test_scene_setup():
            return False
        self._test_primitives()
        if not self._test_simulation():
            return False
        self._test_cinematic_camera()
        self._test_video()
        self._test_physics_accuracy()

        try:
            if self.scene and self.scene._is_setup:
                self.scene.cleanup()
        except Exception:
            pass  # Already disconnected

        print("\n" + "=" * 70)
        print(f"M5 BENCHMARK: {self.passed}/30 PASSED")
        print("=" * 70)

        return self.passed >= 25


if __name__ == "__main__":
    benchmark = M5Benchmark()
    success = benchmark.run_all()
    exit(0 if success else 1)

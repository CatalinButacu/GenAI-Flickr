"""
M4 Motion Generator - Benchmark Tests
=======================================
30 rigorous tests for motion generation.

Run: py tests/benchmark_m4.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


_NO_DATASET       = "no dataset"
_GEN_NOT_READY    = "generator not ready"
_SSM_NOT_READY    = "SSM not ready"
_TEST_ACTION_IDX  = "14. Action index populated"
_TEST_WALK_RUN    = "28. Walk vs Run different"
_TEST_KICK_JUMP   = "29. Kick vs Jump different"


class M4Benchmark:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.generator = None

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
            from src.modules.m4_motion_generator import MotionGenerator, MotionClip  # noqa: F401
            self.test("1. MotionGenerator import", True)
        except ImportError as e:
            self.test("1. MotionGenerator import", False, str(e)[:30])
            return False

        try:
            from src.modules.m4_motion_generator.keyword_retriever import MotionRetriever  # noqa: F401
            from src.modules.m4_motion_generator.ssm_model import SSMMotionModel  # noqa: F401
            self.test("2. Backend classes import", True)
        except ImportError as e:
            self.test("2. Backend classes import", False, str(e)[:30])

        try:
            from src.data import KITMLLoader  # noqa: F401
            self.test("3. KITMLLoader import", True)
        except ImportError as e:
            self.test("3. KITMLLoader import", False, str(e)[:30])

        try:
            from src.modules.m4_motion_generator.ssm import MotionSSM, MambaLayer  # noqa: F401
            self.test("4. SSM modules import", True)
        except ImportError as e:
            self.test("4. SSM modules import", False, str(e)[:30])

        try:
            from src.modules.m4_motion_generator.nn_models import TextToMotionSSM  # noqa: F401
            self.test("5. Training module import", True)
        except ImportError as e:
            self.test("5. Training module import", False, str(e)[:30])

        return True

    def _test_data(self) -> None:
        print("\n[6-10] DATA")
        print("-" * 50)

        kit_exists = os.path.exists("data/KIT-ML")
        self.test("6. KIT-ML dataset exists", kit_exists)

        if not kit_exists:
            for label in ("7. Mean.npy exists", "8. Std.npy exists", "9. Mean shape", "10. Std shape"):
                self.test(label, False, _NO_DATASET)
            return

        mean_exists = os.path.exists("data/KIT-ML/Mean.npy")
        self.test("7. Mean.npy exists", mean_exists)
        std_exists = os.path.exists("data/KIT-ML/Std.npy")
        self.test("8. Std.npy exists", std_exists)

        if mean_exists:
            mean = np.load("data/KIT-ML/Mean.npy")
            self.test("9. Mean shape (251,)", mean.shape == (251,))
        else:
            self.test("9. Mean shape", False)

        if std_exists:
            std = np.load("data/KIT-ML/Std.npy")
            self.test("10. Std shape (251,)", std.shape == (251,))
        else:
            self.test("10. Std shape", False)

    def _test_generator_setup(self) -> None:
        print("\n[11-15] GENERATOR SETUP")
        print("-" * 50)

        try:
            from src.modules.m4_motion_generator import MotionGenerator
            self.generator = MotionGenerator(use_retrieval=True, use_ssm=True)
            self.test("11. Generator initialized", True)
        except Exception as e:
            self.test("11. Generator initialized", False, str(e)[:30])
            self.generator = None

        if not self.generator:
            for label in ("12. Retriever ready", "13. SSM model ready", _TEST_ACTION_IDX, "15. SSM checkpoint exists"):
                self.test(label, False)
            return

        self.test("12. Retriever ready", self.generator.retriever is not None)
        self.test("13. SSM model ready", self.generator.ssm_model is not None)

        if self.generator.retriever:
            n_actions = len(self.generator.retriever.action_index)
            self.test(_TEST_ACTION_IDX, n_actions >= 10, f"{n_actions} actions")
        else:
            self.test(_TEST_ACTION_IDX, False)

        self.test("15. SSM checkpoint exists", os.path.exists("checkpoints/motion_ssm/best_model.pt"))

    def _test_retrieval(self) -> None:
        print("\n[16-22] RETRIEVAL GENERATION")
        print("-" * 50)

        if not (self.generator and self.generator.retriever):
            for i in range(16, 23):
                self.test(f"{i}. Motion test", False, _GEN_NOT_READY)
            return

        actions = [
            ("16. Walk motion", "A person walks forward"),
            ("17. Run motion", "A person runs fast"),
            ("18. Kick motion", "A person kicks a ball"),
            ("19. Jump motion", "A person jumps up"),
            ("20. Turn motion", "A person turns around"),
            ("21. Wave motion", "A person waves hand"),
            ("22. Stand motion", "A person stands still"),
        ]
        for name, prompt in actions:
            try:
                clip = self.generator.generate(prompt, num_frames=60, prefer="retrieval")
                ok = clip is not None and clip.source in ("retrieved", "semantic_retrieved")
                self.test(name, ok, f"{clip.num_frames}f {clip.source}" if clip else "")
            except Exception as e:
                self.test(name, False, str(e)[:30])

    def _test_ssm(self) -> None:
        print("\n[23-27] SSM GENERATION")
        print("-" * 50)

        if not (self.generator and self.generator.ssm_model and self.generator.ssm_model.model):
            for i in range(23, 28):
                self.test(f"{i}. SSM test", False, _SSM_NOT_READY)
            return

        for name, prompt in [("23. SSM walk", "A person walks"), ("24. SSM run", "A person runs"), ("25. SSM kick", "A person kicks")]:
            try:
                clip = self.generator.generate(prompt, num_frames=60, prefer="ssm")
                self.test(name, clip is not None and clip.source == "generated")
            except Exception as e:
                self.test(name, False, str(e)[:30])

        try:
            clip = self.generator.generate("test", num_frames=100, prefer="ssm")
            self.test("26. Motion shape (100, 251)", clip.frames.shape == (100, 251))
        except Exception as e:
            self.test("26. Motion shape", False, str(e)[:30])

        try:
            clip = self.generator.generate("test", prefer="ssm")
            self.test("27. FPS = 20", clip.fps == 20)
        except Exception as e:
            self.test("27. FPS", False, str(e)[:30])

    def _test_semantic(self) -> None:
        print("\n[28-30] SEMANTIC DIFFERENTIATION")
        print("-" * 50)

        if not self.generator:
            self.test(_TEST_WALK_RUN, False)
            self.test(_TEST_KICK_JUMP, False)
            self.test("30. Action coverage", False)
            return

        try:
            walk = self.generator.generate("walk forward", prefer="retrieval")
            run  = self.generator.generate("run fast", prefer="retrieval")
            diff = abs(np.std(walk.frames) - np.std(run.frames))
            self.test(_TEST_WALK_RUN, diff > 0.1, f"diff={diff:.2f}")
        except Exception as e:
            self.test(_TEST_WALK_RUN, False, str(e)[:30])

        try:
            kick = self.generator.generate("kick", prefer="retrieval")
            jump = self.generator.generate("jump", prefer="retrieval")
            diff = kick.num_frames != jump.num_frames or np.std(kick.frames) != np.std(jump.frames)
            self.test(_TEST_KICK_JUMP, diff)
        except Exception as e:
            self.test(_TEST_KICK_JUMP, False, str(e)[:30])

        try:
            covered = sum(1 for a in ["walk", "run", "kick", "jump", "wave"] if a in self.generator.retriever.action_index)
            self.test("30. Action coverage >= 5", covered >= 5, f"{covered} actions")
        except Exception as e:
            self.test("30. Action coverage", False, str(e)[:30])

    def run_all(self):
        print("=" * 70)
        print("M4 MOTION GENERATOR - BENCHMARK SUITE")
        print("=" * 70)

        if not self._test_imports():
            return False
        self._test_data()
        self._test_generator_setup()
        self._test_retrieval()
        self._test_ssm()
        self._test_semantic()

        print("\n" + "=" * 70)
        print(f"M4 BENCHMARK: {self.passed}/30 PASSED")
        print("=" * 70)

        return self.passed >= 25


if __name__ == "__main__":
    benchmark = M4Benchmark()
    success = benchmark.run_all()
    exit(0 if success else 1)

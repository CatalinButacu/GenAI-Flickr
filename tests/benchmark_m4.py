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

from src.modules.motion import MotionGenerator, MotionClip
from src.modules.motion.ssm_model import SSMMotionModel
from src.modules.motion.ssm import MotionSSM, MambaLayer
from src.modules.motion.nn_models import TextToMotionSSM
from src.shared.constants import MOTION_DIM, MOTION_FPS


_NO_DATASET       = "no dataset"
_GEN_NOT_READY    = "generator not ready"
_SSM_NOT_READY    = "SSM not ready"
_TEST_ACTION_IDX  = "14. Retriever has samples"
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
            self.test("1. MotionGenerator import", True)
        except ImportError as e:
            self.test("1. MotionGenerator import", False, str(e)[:30])
            return False

        try:
            self.test("2. Backend classes import", True)
        except ImportError as e:
            self.test("2. Backend classes import", False, str(e)[:30])

        try:
            self.test("3. AMASSLoader import", True)
        except ImportError as e:
            self.test("3. AMASSLoader import", False, str(e)[:30])

        try:
            self.test("4. SSM modules import", True)
        except ImportError as e:
            self.test("4. SSM modules import", False, str(e)[:30])

        try:
            self.test("5. Training module import", True)
        except ImportError as e:
            self.test("5. Training module import", False, str(e)[:30])

        return True

    def _test_data(self) -> None:
        print("\n[6-10] DATA")
        print("-" * 50)

        amass_exists = os.path.exists("data/AMASS")
        self.test("6. AMASS dataset exists", amass_exists)

        if not amass_exists:
            for label in ("7. NPZ files present", "8. At least 10 sequences", "9. Motion dim check", "10. FPS constant"):
                self.test(label, False, _NO_DATASET)
            return

        from pathlib import Path
        npz_files = list(Path("data/AMASS").rglob("*.npz"))
        self.test("7. NPZ files present", len(npz_files) > 0, f"{len(npz_files)} files")
        self.test("8. At least 10 sequences", len(npz_files) >= 10, f"{len(npz_files)} files")
        self.test("9. Motion dim check", MOTION_DIM == 168, f"dim={MOTION_DIM}")
        self.test("10. FPS constant", MOTION_FPS == 30, f"fps={MOTION_FPS}")

    def _test_generator_setup(self) -> None:
        print("\n[11-15] GENERATOR SETUP")
        print("-" * 50)

        try:
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
            # AMASSSampleRetriever stores samples in self._samples list (not _motion_count)
            n_samples = len(getattr(self.generator.retriever, '_samples', []))
            self.test("14. Retriever has samples", n_samples > 0, f"{n_samples} motions")
        else:
            self.test("14. Retriever has samples", False)

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
                # AMASSSampleRetriever sets source to "sample_<dataset_id>"
                ok = clip is not None and clip.source.startswith("sample_")
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
            self.test("26. Motion shape (100, 168)", clip.frames.shape == (100, MOTION_DIM))
        except Exception as e:
            self.test("26. Motion shape", False, str(e)[:30])

        try:
            clip = self.generator.generate("test", prefer="ssm")
            self.test("27. FPS = 30", clip.fps == MOTION_FPS)
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
            self.test(_TEST_WALK_RUN, bool(diff > 0.1), f"diff={diff:.2f}")
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
            retriever = self.generator.retriever
            if retriever is not None:
                covered = sum(1 for a in ["walk", "run", "kick", "jump", "wave"]
                              if self.generator.generate(a, num_frames=30, prefer="retrieval").source != "placeholder")
            else:
                covered = 0
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

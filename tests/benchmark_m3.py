"""
M3 Asset Generator - Benchmark Tests
======================================
15 rigorous tests for 3D mesh generation.

Run: py tests/benchmark_m3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.m3_asset_generator import ModelGenerator
from src.modules.m3_asset_generator.generator import GeneratedModel, ShapEBackend


class M3Benchmark:
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
    
    def run_all(self):
        print("=" * 70)
        print("M3 ASSET GENERATOR - BENCHMARK SUITE")
        print("=" * 70)
        
        # === IMPORTS AND STRUCTURE (1-5) ===
        print("\n[1-5] IMPORTS AND STRUCTURE")
        print("-" * 50)
        
        # Imports
        try:
            from src.modules.m3_asset_generator import ModelGenerator
            self.test("1. ModelGenerator import", True)
        except ImportError as e:
            self.test("1. ModelGenerator import", False, str(e)[:30])
        
        try:
            from src.modules.m3_asset_generator.generator import ShapEBackend, TripoSRBackend
            self.test("2. Backend classes", True)
        except ImportError as e:
            self.test("2. Backend classes", False, str(e)[:30])
        
        # GeneratedModel
        try:
            gm = GeneratedModel(name="test", mesh_path="/test.obj", backend="test")
            self.test("3. GeneratedModel dataclass", gm.name == "test")
        except Exception as e:
            self.test("3. GeneratedModel dataclass", False, str(e)[:30])
        
        # ModelGenerator init
        try:
            mg = ModelGenerator(backend="shap-e", device="cpu")
            self.test("4. ModelGenerator init", mg.backend_name == "shap-e")
        except Exception as e:
            self.test("4. ModelGenerator init", False, str(e)[:30])
        
        # Backend dict
        backends = ModelGenerator.BACKENDS
        self.test("5. Backend registry", "shap-e" in backends)
        
        # === DEPENDENCIES (6-8) ===
        print("\n[6-8] DEPENDENCIES")
        print("-" * 50)
        
        try:
            import diffusers
            self.test("6. Diffusers installed", True)
        except ImportError:
            self.test("6. Diffusers installed", False)
            
        try:
            import torch
            self.test("7. PyTorch installed", True)
        except ImportError:
            self.test("7. PyTorch installed", False)
        
        has_cuda = False
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            self.test("8. CUDA available", has_cuda)
        except:
            self.test("8. CUDA available", False)
        
        # === BACKEND SETUP (9-11) ===
        print("\n[9-11] BACKEND SETUP")
        print("-" * 50)
        
        try:
            self.generator = ModelGenerator(backend="shap-e", device="cuda" if has_cuda else "cpu")
            setup_ok = self.generator.setup()
            self.test("9. Shap-E setup", setup_ok)
        except Exception as e:
            self.test("9. Shap-E setup", False, str(e)[:40])
            self.generator = None
        
        if self.generator and self.generator._is_setup:
            self.test("10. Backend ready flag", self.generator._is_setup)
            self.test("11. Backend instance", self.generator._backend is not None)
        else:
            self.test("10. Backend ready flag", False, "setup failed")
            self.test("11. Backend instance", False, "setup failed")
        
        # === GENERATION TESTS (12-15) ===
        print("\n[12-15] GENERATION TESTS")
        print("-" * 50)
        
        if self.generator and self.generator._is_setup:
            output_dir = "outputs/3d_models/benchmark"
            os.makedirs(output_dir, exist_ok=True)
            
            # Simple object
            print("  Generating 'a red sphere'...")
            try:
                result = self.generator.generate_from_text(
                    prompt="a red sphere",
                    output_dir=output_dir,
                    name="benchmark_sphere",
                    num_inference_steps=24  # Faster for benchmark
                )
                self.test("12. Generate sphere", result is not None and os.path.exists(result.mesh_path))
            except Exception as e:
                self.test("12. Generate sphere", False, str(e)[:40])
            
            # Check file size
            if result and os.path.exists(result.mesh_path):
                size = os.path.getsize(result.mesh_path)
                self.test("13. Mesh file size > 0", size > 1000, f"size={size}")
            else:
                self.test("13. Mesh file size > 0", False)
            
            # Check mesh path
            if result:
                self.test("14. Mesh path valid", result.mesh_path.endswith(('.obj', '.ply')))
            else:
                self.test("14. Mesh path valid", False)
            
            # Backend label
            if result:
                self.test("15. Backend label", result.backend == "shap-e")
            else:
                self.test("15. Backend label", False)
        else:
            self.test("12. Generate sphere", False, "generator not ready")
            self.test("13. Mesh file size > 0", False, "generator not ready")
            self.test("14. Mesh path valid", False, "generator not ready")
            self.test("15. Backend label", False, "generator not ready")
        
        # === SUMMARY ===
        print("\n" + "=" * 70)
        print(f"M3 BENCHMARK: {self.passed}/15 PASSED")
        print("=" * 70)
        
        return self.passed >= 12


if __name__ == "__main__":
    benchmark = M3Benchmark()
    success = benchmark.run_all()
    exit(0 if success else 1)

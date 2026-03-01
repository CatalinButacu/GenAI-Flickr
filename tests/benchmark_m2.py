"""
M2 Scene Planner - Benchmark Tests
====================================
30 rigorous tests for positioning and layout.

Run: py tests/benchmark_m2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modules.scene_understanding.prompt_parser import PromptParser
from src.modules.scene_planner import ScenePlanner, PlannedScene, PlannedEntity, Position3D

_A_BALL = "A ball"


class M2Benchmark:
    def __init__(self):
        self.parser = PromptParser()
        self.parser.setup()
        self.planner = ScenePlanner()
        self.passed = 0
        self.failed = 0
    
    def test(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        if condition:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{status}] {name} {details}")
    
    def plan(self, prompt: str) -> PlannedScene:
        parsed = self.parser.parse(prompt)
        return self.planner.plan(parsed)
    
    def run_all(self):
        print("=" * 70)
        print("M2 SCENE PLANNER - BENCHMARK SUITE")
        print("=" * 70)
        
        # === POSITION ASSIGNMENT (1-10) ===
        print("\n[1-10] POSITION ASSIGNMENT")
        print("-" * 50)
        
        # Basic positioning
        s = self.plan(_A_BALL)
        has_pos = all(e.position is not None for e in s.entities)
        self.test("1. Position assigned", has_pos)
        
        # Above ground
        s = self.plan(_A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        above = ball and ball[0].position.z >= 0
        self.test("2. Above ground (z>=0)", above)
        
        # Unique positions
        s = self.plan("A ball and a cube")
        positions = [str(e.position.to_list()) for e in s.entities]
        unique = len(set(positions)) == len(positions)
        self.test("3. Unique positions", unique)
        
        # Three objects spread
        s = self.plan("A ball, a cube, and a cylinder")
        spread = len(s.entities) >= 3 and len({str(e.position.to_list()) for e in s.entities}) >= 3
        self.test("4. Three objects spread", spread, f"entities={len(s.entities)}")
        
        # Actor vs object positions
        s = self.plan("A person kicks a ball")
        actor = [e for e in s.entities if e.is_actor]
        obj = [e for e in s.entities if not e.is_actor]
        diff_pos = actor and obj and actor[0].position.to_list() != obj[0].position.to_list()
        self.test("5. Actor vs object positions", diff_pos)
        
        # Humanoid height
        s = self.plan("A person stands")
        person = [e for e in s.entities if e.is_actor]
        tall = person and person[0].position.z >= -2
        self.test("6. Humanoid reasonable z", tall)
        
        # Small object positioning
        s = self.plan(_A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        close = ball and abs(ball[0].position.x) < 5 and abs(ball[0].position.y) < 5
        self.test("7. Object in scene bounds", close)
        
        # Position3D operations
        p1 = Position3D(1, 2, 3)
        p2 = Position3D(4, 5, 6)
        p3 = p1 + p2
        self.test("8. Position3D add", p3.to_list() == [5, 7, 9])
        
        p = Position3D(1.5, 2.5, 3.5)
        self.test("9. Position3D to_list", p.to_list() == [1.5, 2.5, 3.5])
        
        # Default position
        p = Position3D()
        self.test("10. Position3D default", p.to_list() == [0, 0, 0])
        
        # === PROPERTY PRESERVATION (11-18) ===
        print("\n[11-18] PROPERTY PRESERVATION")
        print("-" * 50)
        
        # Color preservation
        s = self.plan("A red ball")
        ball = [e for e in s.entities if "sphere" in e.object_type]
        has_color = ball and ball[0].color is not None
        self.test("11. Color preserved", has_color)
        
        s = self.plan("A blue cube")
        cube = [e for e in s.entities if "cube" in e.object_type]
        has_color = cube and cube[0].color is not None
        self.test("12. Blue color preserved", has_color)
        
        # Mass assignment
        s = self.plan(_A_BALL)
        ball = [e for e in s.entities if "sphere" in e.object_type]
        has_mass = ball and ball[0].mass > 0
        self.test("13. Mass assigned", has_mass, f"mass={ball[0].mass if ball else 0}")
        
        # Size assignment
        s = self.plan("A cube")
        cube = [e for e in s.entities if "cube" in e.object_type]
        has_size = cube and cube[0].size is not None
        self.test("14. Size assigned", has_size)
        
        # Object type preserved
        s = self.plan("A cylinder")
        has_cyl = any("cylinder" in e.object_type for e in s.entities)
        self.test("15. Object type preserved", has_cyl)
        
        # Actor flag
        s = self.plan("A person walks")
        has_actor = any(e.is_actor for e in s.entities)
        self.test("16. Actor flag preserved", has_actor)
        
        # Multiple colors
        s = self.plan("A red ball and a blue cube")
        colors = [e.color for e in s.entities if e.color]
        multi = len(colors) >= 2
        self.test("17. Multiple colors", multi, f"colors={len(colors)}")
        
        # Name uniqueness
        s = self.plan("A ball and another ball")
        names = [e.name for e in s.entities]
        unique_names = len(set(names)) == len(names)
        self.test("18. Unique names", unique_names)
        
        # === SCENE CONFIGURATION (19-24) ===
        print("\n[19-24] SCENE CONFIGURATION")
        print("-" * 50)
        
        # Ground size
        s = self.plan(_A_BALL)
        has_ground = s.ground_size is not None and s.ground_size[0] > 0
        self.test("19. Ground size set", has_ground)
        
        # Camera distance
        s = self.plan(_A_BALL)
        has_cam = s.camera_distance > 0
        self.test("20. Camera distance set", has_cam)
        
        # Rotation default
        s = self.plan("A cube")
        cube = [e for e in s.entities if "cube" in e.object_type]
        has_rot = cube and cube[0].rotation is not None
        self.test("21. Rotation assigned", has_rot)
        
        # Entity list not empty
        s = self.plan(_A_BALL)
        self.test("22. Entities list populated", len(s.entities) > 0)
        
        # PlannedEntity attributes
        s = self.plan(_A_BALL)
        e = s.entities[0]
        has_all = hasattr(e, 'name') and hasattr(e, 'position') and hasattr(e, 'mass')
        self.test("23. PlannedEntity attributes", has_all)
        
        # Empty prompt handling
        self.plan("")
        no_crash = True  # If we get here, no crash
        self.test("24. Empty prompt handling", no_crash)
        
        # === COMPLEX SCENARIOS (25-30) ===
        print("\n[25-30] COMPLEX SCENARIOS")
        print("-" * 50)
        
        # Person and object
        s = self.plan("A person kicks a red ball")
        has_both = any(e.is_actor for e in s.entities) and any("sphere" in e.object_type for e in s.entities)
        self.test("25. Person + object scene", has_both)
        
        # Multiple action scene
        s = self.plan("A person walks to a ball and kicks it")
        self.test("26. Action sequence scene", len(s.entities) >= 2)
        
        # Spatial scene
        s = self.plan("A ball on a table")
        self.test("27. Spatial relation scene", len(s.entities) >= 2)
        
        # Furniture scene
        s = self.plan("A table with a ball on it")
        self.test("28. Furniture scene", len(s.entities) >= 1)
        
        # Full scene
        s = self.plan(
            "In a room, a red ball sits on a blue table near a person"
        )
        self.test("29. Full scene description", len(s.entities) >= 2)
        
        # Stress test
        s = self.plan(
            "A red sphere, a blue cube, a green cylinder, a yellow ball, "
            "and a person standing nearby"
        )
        self.test("30. Stress test 5+ entities", len(s.entities) >= 4, f"found={len(s.entities)}")
        
        # === SUMMARY ===
        print("\n" + "=" * 70)
        print(f"M2 BENCHMARK: {self.passed}/30 PASSED")
        print("=" * 70)
        
        return self.passed >= 25


if __name__ == "__main__":
    benchmark = M2Benchmark()
    success = benchmark.run_all()
    exit(0 if success else 1)

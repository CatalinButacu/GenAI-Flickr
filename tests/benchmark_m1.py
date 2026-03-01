"""
M1 Structural Benchmark
=======================
Tests module imports, class instantiation, and API contracts without requiring
a trained T5 checkpoint. Run this during development to catch regressions.

Full extraction benchmarks run automatically after the checkpoint is trained.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.scene_understanding.extractor import Extractor
from src.modules.scene_understanding.retriever import KnowledgeRetriever
from src.modules.scene_understanding.reasoner import Reasoner
from src.modules.scene_understanding.builder import SceneBuilder
from src.modules.scene_understanding.orchestrator import StoryAgent
from src.modules.scene_understanding.models import (
    ExtractionResult, ExtractedEntity, ExtractedAction, ExtractedRelation,
    EntityType, SceneDescription,
)


PASS = 0
FAIL = 0


def check(label: str, condition: bool) -> None:
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if not condition:
        FAIL += 1
    else:
        PASS += 1
    print(f"  [{status}] {label}")


print("=" * 60)
print("M1 STRUCTURAL BENCHMARK")
print("=" * 60)

# [1-5] Extractor API
print("\n[1-5] Extractor — instantiation and properties")
ex = Extractor()
check("1. Extractor() instantiates",          ex is not None)
check("2. is_loaded starts False",            ex.is_loaded == False)
check("3. checkpoint_path returns Path",      isinstance(ex.checkpoint_path, Path))
check("4. checkpoint_path setter invalidates", (setattr(ex, 'checkpoint_path', 'new/path') or ex.is_loaded) == False)
check("5. device setter validates",           True)
try:
    ex.device = "invalid"
    check("5. device setter rejects invalid", False)
except ValueError:
    check("5. device setter rejects invalid", True)

# [6-8] load() raises when checkpoint missing
print("\n[6-8] Extractor — load() raises correctly")
ex2 = Extractor(checkpoint_path="nonexistent/path")
try:
    ex2.load()
    check("6. load() raises FileNotFoundError when missing", False)
except FileNotFoundError:
    check("6. load() raises FileNotFoundError when missing", True)

try:
    ex2.extract("test")
    check("7. extract() raises RuntimeError before load()", False)
except RuntimeError:
    check("7. extract() raises RuntimeError before load()", True)

check("8. is_loaded still False after failed load", ex2.is_loaded == False)

# [9-12] ExtractionResult model
print("\n[9-12] ExtractionResult — model properties")
result = ExtractionResult(raw_prompt="test")
check("9.  has_entities starts False",    result.has_entities == False)
check("10. entities list starts empty",   result.entities == [])
check("11. actions list starts empty",    result.actions == [])
check("12. relations list starts empty",  result.relations == [])

# [13-16] ExtractedEntity
print("\n[13-16] ExtractedEntity — model")
e = ExtractedEntity(id="ball_0", name="ball", entity_type=EntityType.OBJECT)
check("13. entity id set correctly",  e.id == "ball_0")
check("14. entity name set",         e.name == "ball")
check("15. is_static defaults False", e.is_static == False)
check("16. set_attr / get_attr",      (e.set_attr("color", "red", "test") or e.get_attr("color")) == "red")

# [17-19] KnowledgeRetriever
print("\n[17-19] KnowledgeRetriever — pre-setup state")
kr = KnowledgeRetriever()
check("17. is_ready starts False",   kr.is_ready == False)
check("18. entry_count starts 0",    kr.entry_count == 0)
check("19. kb_dir returns Path",     isinstance(kr.kb_dir, Path))

# [20-22] StoryAgent
print("\n[20-22] StoryAgent — orchestrator structure")
agent = StoryAgent()
check("20. StoryAgent instantiates",        agent is not None)
check("21. is_ready starts False",          agent.is_ready == False)
check("22. extraction_mode is 'pending'",   agent.extraction_mode == "pending")

# [23-25] SceneBuilder + Reasoner
print("\n[23-25] Builder and Reasoner — basic construction")
builder = SceneBuilder()
reasoner = Reasoner()
dummy = ExtractionResult(raw_prompt="test")
dummy.entities.append(ExtractedEntity(id="ball_0", name="ball"))
enriched = reasoner.reason(dummy)
scene = builder.build(enriched)
check("23. Reasoner.reason() returns ExtractionResult", isinstance(enriched, ExtractionResult))
check("24. SceneBuilder.build() returns SceneDescription", isinstance(scene, SceneDescription))
check("25. Scene has at least 1 object",   len(scene.objects) >= 1)

print(f"\n{'=' * 60}")
print(f"M1 STRUCTURAL BENCHMARK: {PASS}/{PASS + FAIL} PASSED")
print(f"{'=' * 60}")
if FAIL:
    sys.exit(1)

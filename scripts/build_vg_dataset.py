"""
VG Dataset Builder (Single Responsibility: Data Transformation)
===============================================================
Converts Visual Genome scene_graphs.json into supervised training pairs
for fine-tuning flan-t5-small on scene extraction.

Each output sample:
  input : "extract scene: <natural language sentence about the scene>"
  target: '{"entities": [...], "actions": [...], "relations": [...]}'

Usage:
    python scripts/build_vg_dataset.py
    python scripts/build_vg_dataset.py --max-samples 10000 --output data/m1_training
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output data structures (mirrors ExtractionResult schema)
# ---------------------------------------------------------------------------

@dataclass
class TrainEntity:
    id: str
    name: str
    type: str = "object"          # object / person / animal / vehicle
    count: int = 1
    attributes: Dict[str, str] = field(default_factory=dict)  # color, material

@dataclass
class TrainAction:
    verb: str
    actor: str
    target: Optional[str] = None

@dataclass
class TrainRelation:
    subject: str
    predicate: str
    object: str                   # noqa: A003 (shadows built-in, intentional for JSON key)

@dataclass
class TrainSample:
    input: str                    # T5 input string
    target: str                   # T5 target JSON string


# ---------------------------------------------------------------------------
# VG schema wrappers  (SRP: "read raw VG data")
# ---------------------------------------------------------------------------

class VGSceneGraph:
    """Thin wrapper around a single VG scene graph dict."""

    # Attributes we care about in scene understanding
    ATTRIBUTE_KEYS = {"color", "material", "size", "shape", "texture", "pattern"}

    def __init__(self, data: dict) -> None:
        self._d = data

    @property
    def image_id(self) -> int:
        return self._d.get("image_id", 0)

    @property
    def objects(self) -> List[dict]:
        return self._d.get("objects", [])

    @property
    def relationships(self) -> List[dict]:
        return self._d.get("relationships", [])

    def get_object_name(self, obj: dict) -> str:
        names = obj.get("names", [])
        return names[0].strip().lower() if names else obj.get("name", "unknown").lower()

    def get_object_attributes(self, obj: dict) -> Dict[str, str]:
        """Extract color/material/size attributes from raw VG attribute lists."""
        attrs: Dict[str, str] = {}
        for attr in obj.get("attributes", []):
            a = attr.strip().lower()
            # Check if it looks like a color word
            if a in {"red", "green", "blue", "yellow", "white", "black",
                     "gray", "grey", "brown", "orange", "purple", "pink"}:
                attrs["color"] = a
            elif a in {"wooden", "wood", "metal", "metallic", "plastic",
                       "glass", "stone", "rubber", "fabric", "leather"}:
                attrs["material"] = re.sub(r"en$", "", a)  # wooden → wood
        return attrs


# ---------------------------------------------------------------------------
# Prompt generator  (SRP: "build natural language sentence from VG graph")
# ---------------------------------------------------------------------------

class PromptGenerator:
    """Generates a descriptive English sentence from a VG scene graph."""

    RELATION_TEMPLATES = {
        "on":           "{s} is on {o}",
        "on top of":    "{s} is on top of {o}",
        "above":        "{s} is above {o}",
        "under":        "{s} is under {o}",
        "below":        "{s} is below {o}",
        "next to":      "{s} is next to {o}",
        "beside":       "{s} is beside {o}",
        "near":         "{s} is near {o}",
        "in front of":  "{s} is in front of {o}",
        "behind":       "{s} is behind {o}",
        "left of":      "{s} is to the left of {o}",
        "right of":     "{s} is to the right of {o}",
        "holding":      "{s} is holding {o}",
        "wearing":      "{s} is wearing {o}",
        "has":          "{s} has {o}",
        "eating":       "{s} is eating {o}",
        "riding":       "{s} is riding {o}",
        "sitting on":   "{s} is sitting on {o}",
        "standing on":  "{s} is standing on {o}",
        "walking on":   "{s} is walking on {o}",
        "looking at":   "{s} is looking at {o}",
        "carrying":     "{s} is carrying {o}",
    }

    def generate(self, graph: VGSceneGraph, obj_id_to_name: Dict[int, str]) -> str:
        """Build a sentence like: 'A person holds a ball. The ball is on a table.'"""
        sentences: List[str] = []
        mentioned: set = set()

        for rel in graph.relationships[:6]:   # cap at 6 rels per scene
            subj_id = rel.get("subject_id", -1)
            obj_id  = rel.get("object_id", -1)
            predicate = rel.get("predicate", "").strip().lower()

            s = obj_id_to_name.get(subj_id, "")
            o = obj_id_to_name.get(obj_id, "")
            if not s or not o or not predicate:
                continue

            template = self.RELATION_TEMPLATES.get(predicate)
            if template:
                sentences.append(template.format(s=_indef(s), o=_indef(o)))
            else:
                sentences.append(f"{_indef(s)} {predicate} {_indef(o)}")

            mentioned.add(subj_id)
            mentioned.add(obj_id)

        # Add unmentioned objects as a "There is a ..." sentence
        for obj in graph.objects:
            oid = obj.get("object_id", -1)
            if oid not in mentioned and oid in obj_id_to_name:
                sentences.append(f"There is {_indef(obj_id_to_name[oid])}")
                mentioned.add(oid)
                if len(sentences) >= 5:
                    break

        return ". ".join(sentences) + "." if sentences else ""


def _indef(noun: str) -> str:
    """Prepend 'a' or 'an' to a noun."""
    noun = noun.strip()
    if not noun:
        return noun
    return ("an " if noun[0] in "aeiou" else "a ") + noun


# ---------------------------------------------------------------------------
# Target JSON builder  (SRP: "build structured extraction JSON from VG graph")
# ---------------------------------------------------------------------------

class TargetBuilder:
    """Converts a VG scene graph into the structured JSON the T5 must produce."""

    # VG predicates that correspond to actions/motions (not static spatial rels)
    ACTION_PREDICATES = {
        "holding", "wearing", "eating", "riding", "carrying",
        "walking", "running", "sitting", "standing", "looking at",
        "throwing", "catching", "kicking", "pushing", "pulling",
        "hitting", "touching", "waving", "playing", "driving",
    }

    def build(self, graph: VGSceneGraph, obj_id_to_name: Dict[int, str]) -> str:
        entities: List[dict] = []
        actions:  List[dict] = []
        relations: List[dict] = []

        seen_ids: set = set()

        # Build entities from objects
        for obj in graph.objects:
            oid  = obj.get("object_id", -1)
            name = obj_id_to_name.get(oid, "")
            if not name or oid in seen_ids:
                continue
            seen_ids.add(oid)
            ent = TrainEntity(id=f"obj_{oid}", name=name)
            # Attach attributes
            sg = VGSceneGraph({})  # reuse attribute extractor statically
            ent.attributes = VGSceneGraph(obj).get_object_attributes(obj)
            entities.append(_clean(asdict(ent)))

        # Build actions + relations from relationships
        for rel in graph.relationships:
            pred = rel.get("predicate", "").strip().lower()
            sid  = rel.get("subject_id", -1)
            oid  = rel.get("object_id",  -1)
            s    = obj_id_to_name.get(sid, "")
            o    = obj_id_to_name.get(oid, "")
            if not s or not o:
                continue

            if pred in self.ACTION_PREDICATES:
                actions.append(asdict(TrainAction(
                    verb=pred, actor=f"obj_{sid}", target=f"obj_{oid}"
                )))
            else:
                relations.append(asdict(TrainRelation(
                    subject=f"obj_{sid}", predicate=pred, object=f"obj_{oid}"
                )))

        return json.dumps({
            "entities":  entities[:10],
            "actions":   actions[:5],
            "relations": relations[:8],
        }, ensure_ascii=False)


def _clean(d: dict) -> dict:
    """Remove None and empty fields from a dict."""
    return {k: v for k, v in d.items() if v not in (None, {}, [], "")}


# ---------------------------------------------------------------------------
# Dataset splitter  (SRP: "split samples into train/val/test")
# ---------------------------------------------------------------------------

class DatasetSplitter:
    def __init__(self, train: float = 0.80, val: float = 0.10, seed: int = 42):
        assert abs(train + val + (1 - train - val) - 1.0) < 1e-9
        self._train = train
        self._val   = val
        self._seed  = seed

    def split(self, samples: List[TrainSample]) -> Tuple[
            List[TrainSample], List[TrainSample], List[TrainSample]]:
        rng = random.Random(self._seed)
        data = list(samples)
        rng.shuffle(data)
        n = len(data)
        t = int(n * self._train)
        v = int(n * self._val)
        return data[:t], data[t:t+v], data[t+v:]


# ---------------------------------------------------------------------------
# JSONL writer  (SRP: "write samples to disk")
# ---------------------------------------------------------------------------

class JsonlWriter:
    def write(self, samples: List[TrainSample], path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps({"input": s.input, "target": s.target},
                                   ensure_ascii=False) + "\n")
        return len(samples)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

class VGDatasetBuilder:
    """
    Orchestrates VG → JSONL training pairs pipeline.
    SRP: owns only the pipeline coordination.
    """

    def __init__(self, vg_dir: Path, output_dir: Path, max_samples: int = 50_000):
        self._vg_dir     = vg_dir
        self._output_dir = output_dir
        self._max        = max_samples
        self._prompt_gen = PromptGenerator()
        self._target_bld = TargetBuilder()
        self._splitter   = DatasetSplitter()
        self._writer     = JsonlWriter()

    def run(self) -> None:
        log.info("Loading scene_graphs.json (~739 MB) — this may take 30–60 s …")
        sg_path = self._vg_dir / "scene_graphs.json"
        if not sg_path.exists():
            log.error("scene_graphs.json not found at %s", sg_path)
            sys.exit(1)

        with open(sg_path, encoding="utf-8") as f:
            raw = json.load(f)

        log.info("Loaded %d scene graphs. Building samples …", len(raw))

        samples: List[TrainSample] = []
        skipped = 0

        for entry in raw:
            if len(samples) >= self._max:
                break
            graph = VGSceneGraph(entry)

            # Build object-id → canonical name map
            obj_id_map: Dict[int, str] = {}
            for obj in graph.objects:
                name = graph.get_object_name(obj)
                if name and name != "unknown":
                    obj_id_map[obj.get("object_id", -1)] = name

            if not obj_id_map:
                skipped += 1
                continue

            prompt_text = self._prompt_gen.generate(graph, obj_id_map)
            if not prompt_text or len(prompt_text) < 10:
                skipped += 1
                continue

            target_json = self._target_bld.build(graph, obj_id_map)

            # Validate JSON
            try:
                parsed = json.loads(target_json)
                if not parsed.get("entities"):
                    skipped += 1
                    continue
            except json.JSONDecodeError:
                skipped += 1
                continue

            samples.append(TrainSample(
                input=f"extract scene: {prompt_text}",
                target=target_json,
            ))

        log.info("Built %d valid samples (%d skipped).", len(samples), skipped)

        train, val, test = self._splitter.split(samples)

        n_tr  = self._writer.write(train, self._output_dir / "train.jsonl")
        n_val = self._writer.write(val,   self._output_dir / "val.jsonl")
        n_te  = self._writer.write(test,  self._output_dir / "test.jsonl")

        log.info("Wrote train=%d  val=%d  test=%d  →  %s", n_tr, n_val, n_te, self._output_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build VG T5 training dataset")
    p.add_argument("--vg-dir",      default="data/M1_VisualGenome",
                   help="Directory containing VG JSON files")
    p.add_argument("--output",      default="data/m1_training",
                   help="Output directory for JSONL splits")
    p.add_argument("--max-samples", type=int, default=50_000,
                   help="Maximum training samples to generate (default: 50 000)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    builder = VGDatasetBuilder(
        vg_dir=Path(args.vg_dir),
        output_dir=Path(args.output),
        max_samples=args.max_samples,
    )
    builder.run()

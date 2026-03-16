"""Build a physical object knowledge base from ConceptNet 5.7 CSV dump.

Downloads the ConceptNet assertions CSV (~550MB compressed), filters to
English physical objects with relevant relations, and produces JSON files
compatible with the KnowledgeRetriever.

Usage:
    python scripts/build_conceptnet_kb.py [--max-entities 50000]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONCEPTNET_URL = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
DOWNLOAD_DIR = Path("data/conceptnet_cache")
KB_OUTPUT_DIR = Path("data/knowledge_base/objects")

# Relations useful for physical object knowledge
RELEVANT_RELATIONS = {
    "/r/IsA",            # cat IsA animal
    "/r/PartOf",         # wheel PartOf car
    "/r/HasA",           # car HasA engine
    "/r/MadeOf",         # table MadeOf wood
    "/r/HasProperty",    # ball HasProperty round
    "/r/UsedFor",        # chair UsedFor sitting
    "/r/AtLocation",     # book AtLocation shelf
    "/r/CapableOf",      # dog CapableOf run
    "/r/CreatedBy",      # bread CreatedBy baking
    "/r/RelatedTo",      # cup RelatedTo mug
}

# Categories for classifying objects
CATEGORY_ISA_KEYWORDS = {
    "furniture": ["furniture", "table", "chair", "desk", "bed", "sofa", "couch",
                  "shelf", "cabinet", "drawer", "bench", "stool", "wardrobe"],
    "vehicle": ["vehicle", "car", "truck", "bus", "bicycle", "motorcycle", "boat",
                "airplane", "train"],
    "tool": ["tool", "hammer", "screwdriver", "wrench", "pliers", "saw", "drill"],
    "container": ["container", "box", "bag", "basket", "bottle", "cup", "mug",
                  "jar", "pot", "pan", "bowl", "bucket", "barrel", "can"],
    "food": ["food", "fruit", "vegetable", "meat", "bread", "cake", "snack"],
    "animal": ["animal", "mammal", "bird", "fish", "insect", "reptile", "pet",
               "dog", "cat"],
    "clothing": ["clothing", "shirt", "pants", "dress", "shoes", "hat", "jacket",
                 "coat"],
    "electronic": ["electronic", "computer", "phone", "television", "camera",
                   "radio", "speaker"],
    "sport_equipment": ["ball", "bat", "racket", "goal", "net"],
    "building_part": ["door", "window", "wall", "floor", "ceiling", "roof",
                      "stair", "column"],
}

# Materials mapping from HasProperty/MadeOf
MATERIAL_KEYWORDS = {
    "wood": ["wood", "wooden", "timber", "oak", "pine", "bamboo"],
    "metal": ["metal", "metallic", "steel", "iron", "aluminum", "copper", "brass"],
    "plastic": ["plastic", "synthetic", "polymer", "nylon"],
    "glass": ["glass", "crystal", "transparent"],
    "fabric": ["fabric", "cloth", "cotton", "silk", "wool", "leather", "textile"],
    "stone": ["stone", "concrete", "marble", "granite", "ceramic", "clay", "brick"],
    "rubber": ["rubber", "latex", "silicone"],
    "paper": ["paper", "cardboard"],
}


def download_conceptnet(force: bool = False) -> Path:
    """Download ConceptNet CSV dump if not cached."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DOWNLOAD_DIR / "conceptnet-assertions-5.7.0.csv.gz"

    if gz_path.exists() and not force:
        log.info("ConceptNet dump already cached: %s (%.0f MB)",
                 gz_path, gz_path.stat().st_size / 1e6)
        return gz_path

    log.info("Downloading ConceptNet dump from %s ...", CONCEPTNET_URL)
    urllib.request.urlretrieve(CONCEPTNET_URL, gz_path, _download_progress)
    print()  # newline after progress
    log.info("Downloaded: %s (%.0f MB)", gz_path, gz_path.stat().st_size / 1e6)
    return gz_path


def _download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(100.0, downloaded / total_size * 100) if total_size > 0 else 0
    mb = downloaded / 1e6
    print(f"\r  {mb:.0f} MB ({pct:.1f}%)", end="", flush=True)


def extract_en_physical_objects(gz_path: Path, max_entities: int) -> dict:
    """Parse the ConceptNet CSV and extract English physical object facts.

    Returns a dict of entity_name → {relations: [...], properties: [...]}
    """
    entities: dict[str, dict] = defaultdict(lambda: {
        "relations": [], "isa": set(), "parts": set(),
        "materials": set(), "properties": set(), "used_for": set(),
        "located_at": set(),
    })

    log.info("Parsing ConceptNet CSV (filtering English physical concepts)...")
    line_count = 0
    kept_count = 0

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            line_count += 1
            if line_count % 5_000_000 == 0:
                log.info("  processed %dM lines, %d relevant facts, %d entities",
                         line_count // 1_000_000, kept_count, len(entities))

            if len(row) < 5:
                continue

            relation = row[1]
            if relation not in RELEVANT_RELATIONS:
                continue

            start = row[2]
            end = row[3]

            # Filter to English only
            if not start.startswith("/c/en/") or not end.startswith("/c/en/"):
                continue

            # Extract clean entity names
            start_name = _clean_concept(start)
            end_name = _clean_concept(end)

            if not start_name or not end_name:
                continue
            if len(start_name) > 50 or len(end_name) > 50:
                continue
            # Skip multi-word phrases (>3 words) to focus on specific objects
            if start_name.count(" ") > 2 or end_name.count(" ") > 2:
                continue

            _record_relation(entities, relation, start_name, end_name)
            kept_count += 1

            if len(entities) >= max_entities * 3:
                break

    log.info("Parsed %d lines, %d relevant facts, %d raw entities",
             line_count, kept_count, len(entities))
    return entities


def _clean_concept(uri: str) -> str:
    """Extract clean name from ConceptNet URI like /c/en/table/n."""
    parts = uri.split("/")
    if len(parts) >= 4:
        return parts[3].replace("_", " ").strip().lower()
    return ""


def _record_relation(entities, relation, start_name, end_name):
    """Store a fact into the appropriate entity bucket."""
    e = entities[start_name]

    if relation == "/r/IsA":
        e["isa"].add(end_name)
    elif relation == "/r/PartOf":
        entities[end_name]["parts"].add(start_name)
    elif relation == "/r/HasA":
        e["parts"].add(end_name)
    elif relation == "/r/MadeOf":
        e["materials"].add(end_name)
    elif relation == "/r/HasProperty":
        e["properties"].add(end_name)
    elif relation == "/r/UsedFor":
        e["used_for"].add(end_name)
    elif relation == "/r/AtLocation":
        e["located_at"].add(end_name)
    elif relation == "/r/RelatedTo":
        e["relations"].append(end_name)


def classify_category(entity_data: dict) -> str:
    """Classify an entity into a category based on IsA relations."""
    all_text = " ".join(entity_data["isa"]) + " " + " ".join(entity_data.get("properties", set()))
    all_text = all_text.lower()

    for cat, keywords in CATEGORY_ISA_KEYWORDS.items():
        for kw in keywords:
            if kw in all_text:
                return cat

    return "object"


def detect_material(entity_data: dict) -> str:
    """Detect primary material from MadeOf and HasProperty relations."""
    all_text = " ".join(entity_data["materials"]) + " " + " ".join(entity_data.get("properties", set()))
    all_text = all_text.lower()

    for mat, keywords in MATERIAL_KEYWORDS.items():
        for kw in keywords:
            if kw in all_text:
                return mat

    return "unknown"


def filter_physical_objects(entities: dict, max_entities: int) -> list[dict]:
    """Filter and rank entities to keep physical objects (most useful for the pipeline)."""
    scored = []
    for name, data in entities.items():
        # Score: prefer entities with more physical facts
        score = (
            len(data["isa"]) * 2
            + len(data["parts"]) * 3
            + len(data["materials"]) * 4
            + len(data["properties"]) * 1
            + len(data["used_for"]) * 2
        )

        # Bonus for entities with physical category matches
        category = classify_category(data)
        if category != "object":
            score += 5

        # Skip abstract/non-physical concepts
        isa_text = " ".join(data["isa"]).lower()
        if any(abstract in isa_text for abstract in [
            "concept", "idea", "feeling", "emotion", "quality",
            "abstract", "relation", "category", "property",
        ]):
            continue

        if score < 2:
            continue

        scored.append((score, name, data))

    scored.sort(key=lambda x: -x[0])

    result = []
    for score, name, data in scored[:max_entities]:
        category = classify_category(data)
        material = detect_material(data)

        entry = {
            "canonical_name": name,
            "aliases": list(data["isa"])[:5],
            "category": category,
            "typical_dimensions_m": {},
            "typical_mass_kg": 1.0,
            "material": material,
            "parts": list(data["parts"])[:10],
            "mesh_prompt": f"a 3D model of a {name}",
            "used_for": list(data["used_for"])[:5],
            "located_at": list(data["located_at"])[:5],
            "properties": list(data["properties"])[:10],
            "score": score,
        }
        result.append(entry)

    return result


def save_kb(entries: list[dict], output_dir: Path, batch_size: int = 1000) -> None:
    """Save KB entries as JSON files (batched)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        batch_dict = {e["canonical_name"]: e for e in batch}
        filename = f"objects_{i:06d}.json"
        path = output_dir / filename
        path.write_text(json.dumps(batch_dict, indent=2), encoding="utf-8")

    log.info("Saved %d entries in %d files → %s",
             len(entries), (len(entries) + batch_size - 1) // batch_size, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Build ConceptNet knowledge base")
    parser.add_argument("--max-entities", type=int, default=50000,
                        help="Maximum number of entities to keep")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    gz_path = download_conceptnet(force=args.force_download)
    raw_entities = extract_en_physical_objects(gz_path, args.max_entities)
    filtered = filter_physical_objects(raw_entities, args.max_entities)

    log.info("Filtered to %d physical object entities", len(filtered))

    # Show some stats
    categories = defaultdict(int)
    materials = defaultdict(int)
    for e in filtered:
        categories[e["category"]] += 1
        materials[e["material"]] += 1

    log.info("Category distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        log.info("  %s: %d", cat, count)

    log.info("Material distribution:")
    for mat, count in sorted(materials.items(), key=lambda x: -x[1])[:10]:
        log.info("  %s: %d", mat, count)

    # Save
    save_kb(filtered, KB_OUTPUT_DIR)

    # Show sample entries
    log.info("Sample entries:")
    for e in filtered[:5]:
        log.info("  %s (cat=%s, mat=%s, parts=%s, score=%d)",
                 e["canonical_name"], e["category"], e["material"],
                 e["parts"][:3], e["score"])


if __name__ == "__main__":
    main()

# M1 Story Agent v2 — Architecture & Development Plan

## 1. Problem Statement

The current M1 is **keyword-matching over a hardcoded dictionary**.
It cannot handle:

| Prompt | What's Missing |
|--------|----------------|
| "2 people having a barbecue in the front yard" | Who are they (age/gender)? What height is the barbecue grill? What does a "front yard" look like? |
| "2 BMWs racing in a straight line" | Who drives them? What does a driver look like? Lighting conditions? |
| "me and my girlfriend having a romantic dance" | Who is "me"? Heights? Dance style? |

We need M1 to:
1. **Extract** any entity from open-domain text (not just balls and cubes)
2. **Infer** implicit/missing attributes via commonsense reasoning
3. **Ask** clarification questions when information is truly ambiguous
4. **Produce** a complete, physics-ready scene description for downstream modules

All of this must work **offline** — no OpenAI API calls, no cloud dependency.

---

## 2. Proposed Architecture

```
                        ┌──────────────────────────────────┐
                        │         User Prompt              │
                        └──────────┬───────────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │   A. Prompt Analyzer (NLU)       │
                    │   ─ Entity extraction            │
                    │   ─ Action/verb extraction        │
                    │   ─ Relation extraction           │
                    │   ─ Attribute extraction          │
                    │   (Trained Seq2Seq Transformer)   │
                    └──────────────┬───────────────────┘
                                   │  Structured extraction
                    ┌──────────────▼──────────────────┐
                    │   B. Knowledge Retriever (RAG)   │
                    │   ─ Object knowledge base        │
                    │   ─ Dimensions, masses, parts    │
                    │   ─ Typical spatial layouts       │
                    │   (Embedding index + JSON KB)     │
                    └──────────────┬───────────────────┘
                                   │  Enriched entities
                    ┌──────────────▼──────────────────┐
                    │   C. Commonsense Reasoner        │
                    │   ─ Infer missing attributes     │
                    │   ─ Apply physical rules          │
                    │   ─ Resolve pronouns/references   │
                    │   (Rule engine + small classifier)│
                    └──────────────┬───────────────────┘
                                   │  Flags for ambiguity
                    ┌──────────────▼──────────────────┐
                    │   D. Clarification Generator     │
                    │   ─ Detect unanswerable gaps     │
                    │   ─ Generate natural questions    │
                    │   ─ Accept user answers           │
                    │   (Template + trained ranker)     │
                    └──────────────┬───────────────────┘
                                   │  Complete scene
                    ┌──────────────▼──────────────────┐
                    │   E. Scene Composer              │
                    │   ─ Merge all info into           │
                    │     SceneDescription dataclass    │
                    │   ─ Validate physics constraints  │
                    │   ─ Assign positions/sizes/masses │
                    └──────────────────────────────────┘
```

---

## 3. Deep Dive: Each Sub-module

### 3A. Prompt Analyzer — Trained Seq2Seq Transformer

**Goal**: Given arbitrary text, produce structured JSON of entities, actions, relations.

**Approach**: Fine-tune a small T5 model (t5-small, 60M params) as a **text-to-JSON** extractor.

**Why T5 / Seq2Seq?**
- Works offline (single .pt file)
- T5 was designed for text-to-text: input is prompt, output is structured JSON
- At 60M params it runs on CPU in <1s per inference
- You own the weights — no API dependency

**Papers that justify this:**
| Paper | Relevance |
|-------|-----------|
| Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2020) — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683) | The foundational seq2seq architecture. Shows text-to-text framing works for NER, QA, summarization. |
| Paolini et al., "Structured Prediction as Translation: A Seq2Seq Approach for Joint IE" (2021) — [arXiv:2107.09693](https://arxiv.org/abs/2107.09693) | **Key paper.** Shows you can train T5 to output structured information extraction as linearized tuples. Entities + relations from text in one pass. |
| Lu et al., "Unified Structure Generation for Universal Information Extraction" (UIE, 2022) — [arXiv:2203.12277](https://arxiv.org/abs/2203.12277) | Unified IE model based on T5. Extracts entities, relations, events in one model. Directly applicable. |
| Josifoski et al., "GenIE: Generative Information Extraction" (2022) — [arXiv:2112.08340](https://arxiv.org/abs/2112.08340) | Autoregressive IE using constrained decoding. Shows how to enforce valid output structure. |

**Input/Output format:**
```
INPUT:  "2 people having a barbecue in the front yard"
OUTPUT: {
  "entities": [
    {"id": "person_1", "type": "human", "attributes": {"count": 2}},
    {"id": "barbecue_1", "type": "object", "name": "barbecue grill"},
    {"id": "yard_1", "type": "environment", "name": "front yard"}
  ],
  "actions": [
    {"verb": "having_barbecue", "actor": "person_1", "instrument": "barbecue_1", "location": "yard_1"}
  ],
  "relations": [
    {"subject": "person_1", "predicate": "located_in", "object": "yard_1"},
    {"subject": "barbecue_1", "predicate": "located_in", "object": "yard_1"}
  ]
}
```

**Training dataset** (see Section 4).

---

### 3B. Knowledge Retriever (RAG for Objects)

**Goal**: Given an entity name (e.g., "barbecue grill"), retrieve its physical properties: typical dimensions, mass, parts, default pose.

**Approach**: Build a **local knowledge base** (JSON files) + embed with sentence-transformers + FAISS vector index.

**Why not just hardcode?**
- Hardcoding doesn't scale: you can't list every possible object.
- RAG lets you add new objects by dropping JSON files into a folder.
- Embedding search handles synonyms: "BBQ" → "barbecue grill" → properties.

**Knowledge Base schema:**
```json
{
  "barbecue_grill": {
    "canonical_name": "barbecue grill",
    "aliases": ["bbq", "grill", "barbecue", "charcoal grill"],
    "category": "outdoor_equipment",
    "typical_dimensions_m": {"width": 0.6, "depth": 0.5, "height": 0.9},
    "typical_mass_kg": 15.0,
    "material": "metal",
    "parts": ["grill_body", "legs", "lid", "grate"],
    "physics": {"is_static": true, "friction": 0.7},
    "mesh_prompt": "a charcoal barbecue grill, outdoor cooking equipment",
    "common_contexts": ["backyard", "park", "patio"],
    "related_objects": ["tongs", "charcoal", "plate", "chair"]
  }
}
```

**Papers:**
| Paper | Relevance |
|-------|-----------|
| Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP" (RAG, 2020) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | Foundational RAG paper. We adapt the retrieval concept to a local KB instead of Wikipedia. |
| Reimers & Gurevych, "Sentence-BERT" (2019) — [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) | The embedding model we'll use for semantic search over object descriptions. Works offline. |
| Zhu et al., "3D-GRAND: 3D Grounded Scene Understanding" (2024) — [arXiv:2406.05132](https://arxiv.org/abs/2406.05132) | Links language to 3D object properties. Closest work to our object knowledge retrieval. |

**How to build the KB:**
1. Scrape typical object dimensions from sources (e.g., manufacturer specs, Wikipedia infoboxes)
2. Manually curate ~200-500 common objects with their properties
3. Programmatically expand using WordNet/ConceptNet for aliases
4. Embed each entry using `all-MiniLM-L6-v2` (22M params, runs on CPU)
5. At runtime: embed query → FAISS nearest-neighbor → return top match

---

### 3C. Commonsense Reasoner

**Goal**: Fill in unstated-but-inferable information.

**Examples:**
| Prompt says | What we infer | How |
|------------|---------------|-----|
| "2 people" | 2 separate humanoids needed | Count extraction |
| "having a barbecue" | People are standing, grill is at waist height | Activity template |
| "boyfriend and girlfriend" | Male + female, male ~1.78m, female ~1.65m | Gender rule + height distribution |
| "racing" | Inside vehicles, high speed, forward motion | Activity template |
| "romantic dance" | Close proximity, slow movement, one leads | Activity template |

**Approach**: A **hybrid system**:

1. **Rule engine** for deterministic physics (gravity = 9.81, objects need support, etc.)
2. **Activity templates** — pre-defined patterns for common human activities
3. **Small classifier** (optional) to pick the right activity template from verb+context

**Papers:**
| Paper | Relevance |
|-------|-----------|
| Sap et al., "ATOMIC: An Atlas of Machine Commonsense" (2019) — [arXiv:1811.00146](https://arxiv.org/abs/1811.00146) | Knowledge graph of If-Then commonsense. "If person X has a barbecue, then X needs: grill, food, outdoor space." |
| Hwang et al., "COMET: Commonsense Transformers" (2021) — [arXiv:2010.05953](https://arxiv.org/abs/2010.05953) | Generative commonsense model. Given a situation, predicts likely attributes/consequences. Could be fine-tuned for our domain. |
| Speer et al., "ConceptNet 5.5" (2017) — [arXiv:1612.03975](https://arxiv.org/abs/1612.03975) | Massive commonsense knowledge graph. "barbecue IsA outdoor_activity", "grill UsedFor cooking". Free, offline, queryable. |

**Activity template format:**
```python
ACTIVITY_TEMPLATES = {
    "barbecue": {
        "required_objects": ["grill", "food"],
        "actors_pose": "standing",
        "typical_setting": "outdoor",
        "actor_positions": "around_object",  # actors positioned around central object
        "grill_height": 0.9,  # meters
        "inferred_objects": ["tongs", "plate"],  # optional extras
    },
    "dance_romantic": {
        "actor_count": 2,
        "actor_positions": "facing_close",  # ~0.3m apart
        "height_relation": "first_taller",  # convention
        "motion_type": "slow_sway",
        "typical_setting": "indoor_or_outdoor",
    },
    "car_racing": {
        "actors_inside": True,  # actors are seated inside vehicles
        "vehicle_required": True,
        "motion_type": "forward_fast",
        "typical_setting": "road_or_track",
    },
}
```

---

### 3D. Clarification Generator

**Goal**: When information is truly ambiguous and cannot be inferred, ask the user.

**Approach:**
1. After extraction + KB lookup + reasoning, check for **unresolved gaps**
2. Gaps have a **confidence score** — only ask when below threshold
3. Generate questions from templates
4. Accept user responses and merge into scene

**Gap detection rules:**
```python
CLARIFICATION_RULES = [
    # If entity is "person" but gender/age unknown AND it matters for the scene
    {"condition": "entity.type == 'human' and entity.gender is None",
     "question": "Should {name} be male or female?",
     "importance": "medium",  # Only ask if it affects appearance
     "default": "infer_from_context"},  # Try to infer first

    # If entity has no known dimensions and KB lookup failed
    {"condition": "entity.dimensions is None and kb_lookup_failed",
     "question": "How big should the {name} be approximately?",
     "importance": "high",
     "default": "use_category_average"},

    # If count is ambiguous
    {"condition": "entity.count_ambiguous",
     "question": "How many {name}s should there be?",
     "importance": "high",
     "default": None},  # Must ask
]
```

**Papers:**
| Paper | Relevance |
|-------|-----------|
| Rao & Daumé III, "Learning to Ask Good Questions" (2018) — [arXiv:1802.06385](https://arxiv.org/abs/1802.06385) | Trains models to generate clarification questions. Directly applicable. |
| Aliannejadi et al., "Asking Clarifying Questions in Open-Domain Information-Seeking Conversations" (2019) — [arXiv:1907.06554](https://arxiv.org/abs/1907.06554) | Framework for when/what to ask. Taxonomy of clarification need types. |
| Zamani et al., "Generating Clarifying Questions for Information Retrieval" (2020) — [arXiv:2005.11314](https://arxiv.org/abs/2005.11314) | Template + neural approach for question generation. |

---

### 3E. Scene Composer

**Goal**: Take all enriched entities and compose the final `SceneDescription`.

This is mostly deterministic code:
- Assign 3D positions based on spatial relations
- Set masses from KB
- Create physics constraints
- Generate camera motions from scene layout
- Produce the style prompt from context

---

## 4. Dataset Strategy: How to Train the Seq2Seq Extractor

### 4.1 The Core Problem
You need (prompt, structured_json) pairs. Thousands of them.

### 4.2 Three-Phase Dataset Construction

**Phase 1: Synthetic generation (bootstrap)**
Write a **scene grammar** that procedurally generates (prompt, scene_json) pairs.

```python
# Grammar rules
TEMPLATES = [
    "{count} {adj} {object} {action} {prep} {location}",
    "{actor} {action} {object} {prep} {location}",
    "{actor1} and {actor2} {activity} {prep} {location}",
]
# Fill with vocabulary → instant 10k+ examples
```

Advantage: Unlimited data, perfect labels. Disadvantage: Artificial phrasing.

**Phase 2: Paraphrase augmentation**
Take synthetic examples and rephrase them:
- Use a local paraphrase model (e.g., `Vamsi/T5_Paraphrase_Paws`, runs offline)
- Or manual rewriting of 500-1000 key examples

This bridges the gap between template language and real human language.

**Phase 3: Manual annotation (gold standard)**
- Write 200-500 real diverse prompts by hand (like your examples)
- Annotate the structured output manually
- Use as test/validation set and for final fine-tuning

**Papers on synthetic data for IE:**
| Paper | Relevance |
|-------|-----------|
| Josifoski et al., "GenIE" (2022) | Uses constrained generation to create training data for IE |
| Wang et al., "GPT-NER: Named Entity Recognition via Large Language Models" (2023) — [arXiv:2304.10428](https://arxiv.org/abs/2304.10428) | Shows you can bootstrap NER training data from templates + augmentation |
| Ye et al., "ZeroGen: Efficient Zero-shot Learning via Dataset Generation" (2022) — [arXiv:2202.07922](https://arxiv.org/abs/2202.07922) | Generate synthetic datasets for downstream training |

### 4.3 Dataset Size Targets
| Phase | Count | Purpose |
|-------|-------|---------|
| Synthetic | 10,000-50,000 | Pre-train the extraction pattern |
| Paraphrased | 2,000-5,000 | Bridge to natural language |
| Manual gold | 300-500 | Validation + final fine-tune |
| **Total** | **~15,000-55,000** | Sufficient for T5-small fine-tuning |

---

## 5. Training Pipeline

### Step 1: Pre-train T5-small on synthetic scene extraction
```
Input:  "a red ball falls on a wooden table"
Target: {"entities":[{"id":"ball_1","type":"sphere","color":"red"},
                     {"id":"table_1","type":"furniture","subtype":"table","material":"wood"}],
         "actions":[{"verb":"fall","actor":"ball_1","target":"table_1"}],
         "relations":[{"subject":"ball_1","predicate":"on","object":"table_1"}]}
```

### Step 2: Fine-tune on paraphrased + manual data
### Step 3: Evaluate on held-out manual set
### Step 4: Export model → `checkpoints/scene_extractor/best_model.pt`

**Training hyperparams (T5-small):**
- Batch size: 16-32
- Learning rate: 3e-4  
- Epochs: 10-30 (with early stopping)
- Max input length: 128 tokens
- Max output length: 512 tokens
- Hardware: Single GPU (RTX 3060 or better), or CPU if patient (~10x slower)

---

## 6. Object Knowledge Base Construction

### 6.1 Sources for Object Data
| Source | What it gives | How to use |
|--------|--------------|------------|
| Wikipedia Infoboxes | Dimensions, weights of common objects | Parse HTML infoboxes |
| ShapeNet metadata | 3D model categories + bounding boxes | Map to our object schema |
| ConceptNet 5.5 | Semantic relations (IsA, HasA, UsedFor) | Enrich object knowledge |
| Common Objects in Context (COCO) | Object categories + sizes | Statistical dimensions |
| Manual curation | Domain-specific objects (BBQ, dance floor) | Hand-label ~200 entries |

### 6.2 KB File Structure
```
data/
  knowledge_base/
    objects/
      outdoor_equipment.json   # BBQ, tent, lawn chair...
      vehicles.json            # car, BMW, bicycle...
      furniture.json           # table, chair, bed...
      humans.json              # default heights, poses...
      sports.json              # ball, racket, goal...
    activities/
      cooking.json             # barbecue, frying...
      sports.json              # racing, playing...
      social.json              # dancing, conversation...
    environments/
      outdoor.json             # yard, park, street...
      indoor.json              # room, kitchen, gym...
    embeddings/
      object_index.faiss       # Pre-built FAISS index
      object_metadata.json     # id → object mapping
```

---

## 7. Development Task List (Ordered)

### Phase A: Foundation (Data + KB) — ~2 weeks
```
A1. Design final output schema (SceneDescription v2 dataclasses)
A2. Build the scene grammar for synthetic dataset generation
A3. Write the synthetic data generator script → 10k+ examples
A4. Manually annotate 100 gold-standard prompts (diverse scenarios)
A5. Create the object knowledge base structure
A6. Curate initial 200 objects with properties
A7. Build activity templates for 20 common activities
```

### Phase B: Prompt Analyzer Model — ~2-3 weeks
```
B1. Set up T5-small fine-tuning pipeline (PyTorch + Hugging Face)
B2. Tokenizer preparation and output format validation
B3. Train on synthetic data (Phase 1)
B4. Paraphrase augmentation pipeline
B5. Fine-tune on augmented data (Phase 2)
B6. Evaluate on gold set, iterate
B7. Export final model checkpoint
B8. Write inference wrapper (load model, run extraction)
```

### Phase C: Knowledge Retriever (RAG) — ~1 week
```
C1. Install sentence-transformers + FAISS
C2. Embed all KB entries with all-MiniLM-L6-v2
C3. Build FAISS index
C4. Write retrieval API: query → top-k objects
C5. Integrate with prompt analyzer output
C6. Test with diverse object queries
```

### Phase D: Commonsense Reasoner — ~1 week
```
D1. Implement rule engine for physics constraints
D2. Build activity template matcher
D3. Implement attribute inference (height from gender, mass from size)
D4. Implement pronoun / reference resolution
D5. Confidence scoring for each inferred attribute
D6. Unit tests with your example prompts
```

### Phase E: Clarification Generator — ~1 week
```
E1. Define gap detection rules
E2. Build question templates
E3. Implement question ranking (importance scoring)
E4. Build interactive clarification loop (ask → answer → merge)
E5. Add "auto-resolve" mode (skip questions, use defaults)
E6. Integration test with pipeline
```

### Phase F: Scene Composer + Integration — ~1 week
```
F1. Design SceneDescription v2 (richer than current)
F2. Position solver (spatial relations → 3D coordinates)
F3. Physics property assignment from KB
F4. Camera motion inference from scene layout
F5. Integration with existing pipeline.py
F6. End-to-end test: prompt → video (through all modules)
```

---

## 8. File Structure (New M1)

```
src/modules/m1_scene_understanding/
    __init__.py                          # Public exports
    agent.py                             # StoryAgent v2 (orchestrator)
    prompt_analyzer/
        __init__.py
        model.py                         # T5-based extractor (inference)
        schema.py                        # Output dataclasses
        train.py                         # Training script
    knowledge_retriever/
        __init__.py
        retriever.py                     # FAISS-based RAG lookup
        embedder.py                      # Sentence-transformer wrapper
        index_builder.py                 # Build FAISS index from KB
    commonsense_reasoner/
        __init__.py
        reasoner.py                      # Main reasoning engine
        rules.py                         # Physics + common sense rules
        activity_templates.py            # Templates for human activities
    clarification/
        __init__.py
        gap_detector.py                  # Find missing information
        question_generator.py            # Generate clarification questions
        answer_merger.py                 # Merge user answers into scene
    scene_composer/
        __init__.py
        composer.py                      # Final scene assembly
        position_solver.py               # Spatial relations → 3D coords
        physics_assigner.py              # Mass, friction, constraints
data/
    knowledge_base/
        objects/                          # Object property JSONs
        activities/                       # Activity template JSONs
        environments/                     # Environment descriptions
        embeddings/                       # FAISS index + metadata
    training/
        synthetic/                        # Generated training data
        augmented/                        # Paraphrased versions
        gold/                             # Hand-annotated test set
scripts/
    generate_training_data.py            # Synthetic dataset generator
    train_prompt_analyzer.py             # T5 fine-tuning script
    build_knowledge_base.py              # KB construction + indexing
    evaluate_extraction.py               # Metrics on gold set
checkpoints/
    scene_extractor/
        best_model.pt                    # Trained T5 weights
        tokenizer/                       # Saved tokenizer
```

---

## 9. Key Dependencies (all offline-capable)

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `transformers` | >=4.30 | T5 model loading, tokenizer | ~500MB with model |
| `torch` | >=2.0 | Training + inference | Already installed |
| `sentence-transformers` | >=2.2 | Embedding for RAG | ~100MB with model |
| `faiss-cpu` | >=1.7 | Vector similarity search | ~20MB |
| `datasets` | >=2.14 | Dataset management | Light |
| `spacy` | >=3.6 | Backup NLP (optional) | ~15MB with en_core_web_sm |

---

## 10. Evaluation Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| Entity Recall | % of entities in gold set that model finds | >85% |
| Entity Precision | % of extracted entities that are correct | >80% |
| Action F1 | Correct verb + actor + target | >75% |
| Relation F1 | Correct spatial/temporal relations | >70% |
| KB Hit Rate | % of entities that get properties from KB | >90% for common objects |
| Inference Accuracy | % of inferred attributes that are reasonable | >80% (human eval) |
| Clarification Relevance | Are generated questions useful? | >85% (human eval) |
| End-to-End | Does the final scene look correct? | Qualitative |

---

## 11. References (Complete)

### Core Architecture
1. Raffel et al., "Exploring the Limits of Transfer Learning with T5" (2020) — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
2. Paolini et al., "Structured Prediction as Translation" (2021) — [arXiv:2107.09693](https://arxiv.org/abs/2107.09693)
3. Lu et al., "Unified Structure Generation for Universal IE (UIE)" (2022) — [arXiv:2203.12277](https://arxiv.org/abs/2203.12277)

### Knowledge & Commonsense
4. Lewis et al., "Retrieval-Augmented Generation (RAG)" (2020) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
5. Sap et al., "ATOMIC: An Atlas of Machine Commonsense" (2019) — [arXiv:1811.00146](https://arxiv.org/abs/1811.00146)
6. Hwang et al., "COMET: Commonsense Transformers" (2021) — [arXiv:2010.05953](https://arxiv.org/abs/2010.05953)
7. Speer et al., "ConceptNet 5.5" (2017) — [arXiv:1612.03975](https://arxiv.org/abs/1612.03975)

### Scene Understanding
8. Tan et al., "Text2Scene" (CVPR 2019) — [arXiv:1809.01110](https://arxiv.org/abs/1809.01110)
9. Chang et al., "Learning Spatial Common Sense" (EMNLP 2018) — Stanford spatial reasoning
10. Zhu et al., "3D-GRAND: 3D Grounded Scene Understanding" (2024) — [arXiv:2406.05132](https://arxiv.org/abs/2406.05132)

### Clarification & Disambiguation
11. Rao & Daumé III, "Learning to Ask Good Questions" (2018) — [arXiv:1802.06385](https://arxiv.org/abs/1802.06385)
12. Aliannejadi et al., "Asking Clarifying Questions" (2019) — [arXiv:1907.06554](https://arxiv.org/abs/1907.06554)

### Dataset Construction
13. Josifoski et al., "GenIE: Generative Information Extraction" (2022) — [arXiv:2112.08340](https://arxiv.org/abs/2112.08340)
14. Ye et al., "ZeroGen: Efficient Zero-shot via Dataset Generation" (2022) — [arXiv:2202.07922](https://arxiv.org/abs/2202.07922)
15. Reimers & Gurevych, "Sentence-BERT" (2019) — [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

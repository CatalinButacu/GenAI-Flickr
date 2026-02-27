# Dissertation Roadmap — Physics-Constrained Video Generation
*Authored: 2026-02-26 | Hardware: RTX 3050 Laptop 4 GB VRAM, CUDA 12*

---

## Current State Snapshot

| Module | Status | Blocking? | Notes |
|--------|--------|-----------|-------|
| **M1** Scene Understanding | 🔄 Training v5 | YES — everything downstream waits for M1 | eval_loss 0.0719 @ epoch 4, target < 0.05 |
| **M2** Scene Planner | ✅ Working (rules-based) | No | Reliable for simple scenes; needs constraint solver for complex |
| **M3** Asset Generator | ⚡ Optional / Shap-E | No | Works when GPU available; acceptable as-is |
| **M4** Motion Generator | ⚡ Optional / retrieval+SSM | No | Retrieval works; SSM trained; semantic matching missing |
| **M5** Physics Engine | ✅ Working / PyBullet | No | Video export works, cinematic camera works |
| **M6** RL Controller | 🔧 Stub | No (optional) | `NotImplementedError`; lowest priority unless required by dissertation |
| **M7** Render Engine | 🔧 Stub | No (optional) | Post-processing placeholder |
| **M8** AI Enhancer | ⚡ Optional / ControlNet | No | Works per-frame; no temporal consistency yet |
| **Tests** | ✅ All passing | No | 62 SonarQube issues fixed; clean |
| **Dataset** | ✅ Clean | No | 50k samples, 0 errors |

**Active process:** `train_m1_t5.py` — v5, step ~10k/37.5k, ~28%, ETA ~6h (runs overnight)

---

## Strategic Phases

```
PHASE 0  ──  Now             Train & validate M1 to production quality
PHASE 1  ──  M1 ready        Improve M2 layout + M4 semantic retrieval
PHASE 2  ──  M2+M4 ready     Full pipeline integration & quality benchmarks
PHASE 3  ──  Pipeline solid  (Optional) M6 RL controller
PHASE 4  ──  Code complete   Dissertation writing & evaluation section
```

---

## PHASE 0 — M1 Training to Production Quality

**Goal:** M1 reaches eval_loss ≤ 0.05, parses arbitrary English prompts into valid
`{entities, actions, spatial_relations}` JSON reliably.

### 0.1 — Monitor current training run

- Training command:
  ```
  python scripts/train_m1_t5.py
      --output m1_checkpoints/m1_scene_extractor_v5
      --resume-from m1_checkpoints/m1_scene_extractor_v5/checkpoint-10000
      --epochs 15
  ```
- **Checkpoints saved every 2,500 steps** → auto-resume if interrupted
- Best checkpoint auto-selected by `load_best_model_at_end=True`
- **Done when:** training exits, check `eval_loss` in `trainer_state.json`

### 0.2 — Post-training evaluation

```bash
python scripts/test_m1_inference.py \
    --model m1_checkpoints/m1_scene_extractor_v5 \
    --prompts "a red ball falls on a blue cube" \
              "a person walks to the table and picks up the book" \
              "two robots collide and fall over" \
              "a dog jumps over a fence near a tree"
```

**Pass criteria:**
- [ ] All 4 prompts → valid JSON (no parse errors)
- [ ] Entities list is non-empty and sensible
- [ ] Actions list contains correct action types
- [ ] Spatial relations captured (e.g., "near a tree" → `NEAR`)
- [ ] eval_loss ≤ 0.05 in `trainer_state.json`

### 0.3 — Wire M1 T5 model into the live pipeline

- File: `src/modules/m1_scene_understanding/orchestrator.py` (or add to `prompt_parser.py`)
- Currently the pipeline uses `PromptParser` (rules-based, no GPU)
- Add a `T5SceneParser` class that loads the fine-tuned model and falls back to `PromptParser` on failure
- `Pipeline.setup()` should prefer `T5SceneParser` when model path exists

```python
# Sketch for src/modules/m1_scene_understanding/t5_parser.py
class T5SceneParser:
    """Inference wrapper for the fine-tuned flan-T5-small M1 model."""
    DEFAULT_MODEL = "m1_checkpoints/m1_scene_extractor_v5"

    def __init__(self, model_path: str = DEFAULT_MODEL):
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self._tokenizer = T5Tokenizer.from_pretrained(model_path)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path).eval()

    def parse(self, prompt: str) -> ParsedScene:
        inputs = self._tokenizer(
            f"extract scene: {prompt}", return_tensors="pt", max_length=256, truncation=True
        )
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=256)
        raw = self._tokenizer.decode(ids[0], skip_special_tokens=False)
        # post-process: replace <extra_id_0>/<extra_id_1> → {/}
        ...
```

**Deliverable:** `python main.py "a red ball falls on a blue cube"` uses the T5 model.

---

## PHASE 1 — Improve M2 Layout + M4 Semantic Retrieval

**Prerequisite:** Phase 0 complete (M1 reliable)

### 1.1 — M2: Constraint-based spatial layout

**Problem:** current `ScenePlanner` hard-codes 3 action types; fails on anything like
"the chair is behind the table" or multi-entity arrangements.

**Plan:** add `ConstraintLayout` class using `scipy.optimize`:

```python
# src/modules/m2_scene_planner/constraint_layout.py
from scipy.optimize import minimize
import numpy as np

RELATION_CONSTRAINTS = {
    "ON":         lambda pa, pb, ha, hb: (pa[2] - pb[2] - (hb + ha) / 2) ** 2,
    "ABOVE":      lambda pa, pb, ha, hb: max(0.0, pb[2] - pa[2] + 0.3) ** 2,
    "BESIDE":     lambda pa, pb, ha, hb: max(0.0, 0.5 - abs(pa[0] - pb[0])) ** 2,
    "IN_FRONT_OF":lambda pa, pb, ha, hb: max(0.0, 0.5 - (pb[1] - pa[1])) ** 2,
    "NEAR":       lambda pa, pb, ha, hb: max(0.0, np.linalg.norm(pa - pb) - 1.0) ** 2,
}
```

- `ScenePlanner.plan()` calls `ConstraintLayout.solve(entities, spatial_relations)` first,
  then falls back to current `_place_actors / _place_objects` if no relations present
- `pip install scipy` (already in most envs)

**Effort:** 1–2 days  
**Test:** `tests/benchmark_m2.py` must still pass + add 5 new spatial relation tests

### 1.2 — M4: Semantic retrieval (SBERT)

**Problem:** `MotionRetriever` matches on exact keywords (`walk`, `run`…); misses
`"stroll"`, `"sprint"`, `"leap"`, `"shuffle"`, etc.

**Plan:** add `SemanticRetriever` subclass:

```python
# src/modules/m4_motion_generator/semantic_retriever.py
from sentence_transformers import SentenceTransformer, util

class SemanticRetriever(MotionRetriever):
    def __init__(self, data_dir="data/KIT-ML"):
        super().__init__(data_dir)
        self._sbert = SentenceTransformer("all-MiniLM-L6-v2")  # 80 MB
        if self._samples:
            self._embs = self._sbert.encode(
                [s.text for s in self._samples], convert_to_tensor=True, show_progress_bar=True
            )

    def retrieve(self, text: str, max_frames: int = 200):
        if not self._samples:
            return None
        q = self._sbert.encode(text, convert_to_tensor=True)
        idx = int(util.cos_sim(q, self._embs)[0].argmax())
        s = self._samples[idx]
        return MotionClip(action=text, frames=s.motion[:max_frames], source="semantic_retrieved")
```

- Make `MotionGenerator` default to `SemanticRetriever` when `backend="retrieval"`
- `pip install sentence-transformers`

**Effort:** 1 day  
**Test:** query `"stroll"` → retrieves a `walk` clip (not random); SBERT similarity > 0.7

---

## PHASE 2 — Full Pipeline Integration & Benchmarks

**Prerequisite:** Phases 0 + 1 complete

### 2.1 — Integration smoke test

Run the full pipeline end-to-end on 10 prompts spanning:
- Object-only (no actor): `"a sphere bounces on a platform"`
- Actor + action: `"a person walks and kicks a ball"`
- Spatial relations: `"a cube is on top of a table"`
- Multi-entity: `"three balls fall onto a flat surface"`
- Complex: `"a person runs, picks up a box, and carries it"`

```bash
python tests/run_all_benchmarks.py
```

All 5 benchmark files (M1–M5) must pass. Capture output JSON.

### 2.2 — Evaluation metrics (dissertation requirement)

Write `tests/evaluate_pipeline.py`:

| Metric | How to measure |
|--------|---------------|
| **M1 parse accuracy** | JSON validity rate + entity/action F1 against ground truth |
| **M1 BLEU** | Already computed in `train_m1_t5.py` → `eval_bleu` |
| **M2 layout validity** | No overlapping entities (AABB check), all positions inside world bounds |
| **M4 motion quality** | FID-like: KL divergence of joint velocity distributions vs. KIT-ML test set |
| **M5 physics accuracy** | Energy conservation: `|E_final/E_initial - 1| < 0.05` (already in `benchmark_m5.py`) |
| **End-to-end** | Manual rating on 20 prompts, 1–5 scale for: realism, text alignment, smoothness |

### 2.3 — Output quality pass

- Check generated MP4s for:
  - [ ] No black frames / rendering artifacts
  - [ ] Camera framing is appropriate
  - [ ] Motion duration matches `--duration` flag
  - [ ] Actors don't clip through floor

---

## PHASE 3 — Optional: M6 RL Controller

**Only pursue if dissertation scope requires a learning-based physics controller.**  
*If not required: skip and note as "future work".*

### 3.1 — Gymnasium environment wrapper

File: `src/modules/m6_rl_controller/physics_env.py`

```python
import gymnasium as gym
import numpy as np

class PhysicsSceneEnv(gym.Env):
    """M5 PhysicsScene wrapped as a standard Gymnasium environment."""
    def __init__(self, scene_config, reference_motion: np.ndarray):
        ...
    # Full implementation in MODULE_RESEARCH.md → M6 section
```

Install: `pip install stable-baselines3 gymnasium`

### 3.2 — PPO training

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, device="cuda", verbose=1)
model.learn(total_timesteps=1_000_000)   # ~3h on RTX 3050
model.save("checkpoints/m6_ppo/walk_v1")
```

**Reward:**
```
R = 0.5 × R_pose_match    # -||q_sim - q_ref||²  (from M4 reference)
  + 0.3 × R_alive          # +1 per step if not fallen
  + 0.1 × R_velocity       # match root velocity
  + 0.1 × R_energy_eff     # -torque²  (smoothness)
```

### 3.3 — Integration

- `Pipeline.run()` already has `use_rl_controller` flag (currently False)
- When True: M6 overrides joint positions produced by M4 with RL-controlled ones
- Fallback: if PPO model not found, revert to M4-only motion

**Effort:** 3–5 days  
**Deliverable:** a physically-plausible walking humanoid controlled by PPO

---

## PHASE 4 — Dissertation Writing

### 4.1 — Chapter → Code mapping

| Chapter | Code evidence to include |
|---------|--------------------------|
| 2. Literature Review | MODULE_RESEARCH.md references |
| 3. Architecture | `src/pipeline.py`, `doc/planning/ARCHITECTURE.md`, module diagram |
| 4. M1 Scene Understanding | `train_m1_t5.py`, dataset stats, eval_loss curves (wandb) |
| 5. Physics Simulation | `src/modules/m5_physics_engine/`, benchmark_m5 results |
| 6. Motion Generation | `src/modules/m4_motion_generator/`, KIT-ML retrieval stats |
| 7. Evaluation | `tests/evaluate_pipeline.py` results, 20-prompt manual ratings |
| 8. Conclusion & Future Work | M6 sketch, M8 AnimateDiff, LayoutGPT roadmap |

### 4.2 — Figures to generate

- [ ] Pipeline architecture diagram (Mermaid → PNG)
- [ ] M1 training loss curves (from wandb or `trainer_state.json`)
- [ ] Sample frame grids: 5 prompts × 3 frames (PyBullet renders)
- [ ] M4 retrieval similarity heatmap (SBERT cosine scores)
- [ ] Evaluation bar chart (per-metric, M1 vs. baseline PromptParser)

### 4.3 — Key claims to validate (need numbers)

1. *"T5-based parsing outperforms keyword rules"* → compare M1 T5 vs. `PromptParser` on 100-prompt held-out set
2. *"Physics simulation produces physically plausible motions"* → energy conservation metric
3. *"Semantic retrieval improves motion relevance"* → SBERT top-1 relevance vs. keyword matching

---

## Timeline Estimate

```
Week 1  (now)     PHASE 0  M1 training completes, T5SceneParser wired into pipeline
Week 2            PHASE 1  M2 ConstraintLayout + M4 SemanticRetriever implemented
Week 3            PHASE 2  Full pipeline integration + evaluation metrics
Week 4            PHASE 2  Quality pass, generate demo videos, fix bugs
Week 5+           PHASE 3  (optional) M6 RL Controller
Final weeks       PHASE 4  Dissertation writing + figure generation
```

---

## Dependency Graph

```
M1 T5 model (Phase 0)
    │
    ├──► T5SceneParser wired        (0.3)
    │
    ├──► M2 ConstraintLayout        (1.1)  ← scipy — no extra model needed
    │         │
    │         └──► Integration test (2.1)
    │
    ├──► M4 SemanticRetriever       (1.2)  ← SBERT all-MiniLM — 80 MB
    │         │
    │         └──► Integration test (2.1)
    │
    └──► Evaluation metrics         (2.2)  ← dissertation requirement
              │
              ├──► (optional) M6 PPO (3.x)  ← stable-baselines3
              │
              └──► Chapter writing  (4.x)
```

---

## Install Checklist (all phases)

```bash
# Phase 0 (already in requirements.txt — confirm)
pip install transformers datasets torch

# Phase 1
pip install scipy sentence-transformers

# Phase 3 (optional)
pip install stable-baselines3 gymnasium mujoco

# Visualization
pip install matplotlib seaborn wandb
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| M1 training stalls / loss plateaus | Low | High | Already at 0.0719 — almost certainly improvable; early stop if no progress by epoch 8 |
| RTX 3050 VRAM OOM during inference | Medium | Medium | Use `torch.float16`, `attention_slicing`, batch_size=1 |
| KIT-ML dataset action coverage gap | Medium | Low | SBERT retrieval bridges unseen verbs; placeholder fallback exists |
| M6 PPO training instability | High | Low | Skip M6 — already planned as optional; note as future work |
| T5 brace tokenization bugs (extra_id) | Medium | Medium | `_sub_braces()` in `train_m1_t5.py` already handles this; post-process to reverse |

---

*See also: `doc/research/MODULE_RESEARCH.md` for per-module library deep-dives.*

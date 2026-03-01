# Physics-Constrained Video Generation

Dissertation project — generate physics-realistic videos from text prompts.

## Quick Start

```bash
pip install -r requirements.txt
python main.py "a red ball falls onto a blue cube"
```

## Pipeline

```
Text Prompt
    ↓ M1  Scene Understanding  — parse entities, actions, spatial relations
    ↓ M2  Scene Planner        — place entities in 3-D world space
    ↓ M3  Asset Generator      — Shap-E 3-D mesh generation    (optional)
    ↓ M4  Motion Generator     — SSM / KIT-ML motion clips     (optional)
    ↓ M5  Physics Engine       — PyBullet simulation + camera
    ↓ M6  RL Controller        — PPO humanoid control          (stub)
    ↓ M7  Render Engine        — post-processing               (stub)
    ↓ M8  AI Enhancer          — ControlNet frame enhancement  (optional)
    ↓ MP4 Video
```

## Usage

```bash
# Default
python main.py "a red ball falls onto a blue cube"

# Custom duration / fps
python main.py "a person walks and kicks a ball" --duration 8 --fps 30

# Enable 3-D mesh generation (requires ~4 GB GPU)
python main.py "a wooden chair tips over" --with-assets

# Enable ControlNet enhancement (requires ~8 GB GPU)
python main.py "a sphere bounces" --with-enhance

# Disable motion generation (faster)
python main.py "boxes collide" --no-motion

python main.py --help
```

## Python API

```python
from src.pipeline import Pipeline, PipelineConfig

config   = PipelineConfig(duration=8, fps=30)
pipeline = Pipeline(config)
result   = pipeline.run("A person walks to a ball and kicks it")
print(result["video_path"])   # outputs/videos/output.mp4
```

## Module Reference

| # | Module | Status | Key files |
|---|--------|--------|-----------|
| M1 | Scene Understanding | ✅ Active | `prompt_parser.py`, `orchestrator.py` |
| M2 | Scene Planner | ✅ Active | `planner.py` |
| M3 | Asset Generator | ⚡ Optional | `generator.py` (Shap-E) |
| M4 | Motion Generator | ⚡ Optional | `generator.py` (SSM / KIT-ML) |
| M5 | Physics Engine | ✅ Active | `scene.py`, `simulator.py` |
| M6 | RL Controller | 🔧 Stub | `controller.py` (PPO planned) |
| M7 | Render Engine | 🔧 Stub | `render_engine.py` (post-proc planned) |
| M8 | AI Enhancer | ⚡ Optional | `renderer.py` (ControlNet) |

## Project Structure

```
main.py                        # Single entry point
src/
├── pipeline.py                # Pipeline orchestrator
├── shared/
│   └── vocabulary.py          # Canonical objects, actions, properties
└── modules/
    ├── scene_understanding/ # M1: Text → ParsedScene
    │   ├── prompt_parser.py   #   Rules-based parser (fast, no GPU)
    │   └── orchestrator.py    #   T5 ML parser (StoryAgent)
    ├── scene_planner/       # M2: ParsedScene → PlannedScene
    ├── asset_generator/     # M3: entity → 3-D mesh (optional)
    ├── motion_generator/    # M4: action text → motion clip (optional)
    ├── physics_engine/      # M5: PlannedScene → frames + video
    ├── m6_rl_controller/       # M6: PPO control (stub)
    ├── render_engine/       # M7: post-processing (stub)
    └── ai_enhancer/         # M8: ControlNet enhance (optional)
config/
└── default.yaml               # Physics / camera / output defaults
scripts/                       # Training utilities (M1 T5, M4 SSM)
tests/                         # Benchmark suites per module
examples/                      # Standalone demo scripts
```

## M1: Two parser modes

| Mode | Class | Speed | Accuracy |
|------|-------|-------|----------|
| Rules (default) | `PromptParser` | Fast, no GPU | Good for common objects/actions |
| ML (T5 seq2seq) | `StoryAgent` | Slow, requires checkpoint | Higher accuracy |

Train the T5 extractor:
```bash
python scripts/train_m1_t5.py
```

## Requirements

- Python 3.10+
- CUDA GPU for M3 (Shap-E) and M8 (ControlNet)
- CPU-only sufficient for M1 (rules), M2, M5

## License

MIT

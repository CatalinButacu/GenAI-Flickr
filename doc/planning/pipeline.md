# Pipeline Orchestrator - Technical Planning

## Module Input/Output

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `str` | User prompt |
| **Output** | `dict` | Contains `video_path`, `scene_description`, `generated_models`, `physics_frames` |

---

## Class: `PipelineConfig` (lines 11-22)
| Field | Default | Purpose |
|-------|---------|---------|
| `output_dir` | "outputs" | Where videos are saved |
| `assets_dir` | "assets/objects" | Where 3D models are saved |
| `use_3d_generation` | True | Enable Shap-E |
| `use_controlnet` | False | Enable ControlNet (5GB download) |
| `use_llm` | False | Use GPT for parsing |
| `fps` | 24 | Output video framerate |
| `duration` | 5.0 | Simulation length in seconds |
| `device` | "cuda" | GPU device |

---

## Class: `Pipeline`

### `__init__(self, config: PipelineConfig = None)`
Initializes config and nulls all modules.

### `setup(self) -> bool`
Lazy-loads all modules:
1. `StoryAgent` (always)
2. `ModelGenerator` (if `use_3d_generation`)
3. `VideoRenderer` (if `use_controlnet`)

### `run(self, prompt: str, output_name: str) -> dict`
Main orchestration:

| Step | Module | Action |
|------|--------|--------|
| 1/5 | StoryAgent | `parse(prompt)` → SceneDescription |
| 2/5 | ModelGenerator | Generate meshes for "mesh" objects |
| 3/5 | PhysicsEngine | Setup scene + run_cinematic() |
| 4/5 | VideoRenderer | enhance_frames() (if ControlNet) |
| 5/5 | Output | create_video() → MP4 |

### `run_simple(self, prompt, output_name) -> str`
Wrapper that returns only `video_path`.

---

## Full Pipeline Flow

```
                   User Prompt
                       │
                       ▼
               ┌─────────────┐
               │  Pipeline   │
               │   .run()    │
               └─────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
 StoryAgent      ModelGenerator    PhysicsEngine
     │                 │                 │
     └────────────────→│←────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │ VideoRenderer  │  (optional)
              └────────────────┘
                       │
                       ▼
                   output.mp4
```

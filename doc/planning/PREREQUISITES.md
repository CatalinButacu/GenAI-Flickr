# Module Prerequisites & Datasets

This document defines the datasets, prerequisites, and dependencies for each module.

---

## Shared Vocabulary (Core Foundation)

> **All modules share a common vocabulary for actions, objects, and properties.**

### Location

```
src/shared/vocabulary.py
```

### Purpose

- Ensures RL action space matches NLP parser output
- Enables consistent object types across Asset Generator and Physics
- Provides ground truth for validation

---

## Module 1: Prompt Parser

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| spaCy model | `en_core_web_sm` | `pip install spacy && python -m spacy download en_core_web_sm` |
| Shared vocabulary | `src/shared/vocabulary.py` | Must be created first |

### Datasets (Optional - for fine-tuning)

| Dataset | Purpose | Link |
|---------|---------|------|
| Custom prompts | Train your parser | Create manually (50-100 examples) |

### Paper References

- Scene graph parsing: [Chang et al., 2018](https://cs.stanford.edu/~pliang/papers/spatial-emnlp2018.pdf)

---

## Module 2: Scene Planner

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| Module 1 output | Parsed prompt → JSON | Depends on Module 1 |
| Spatial priors | Heuristics or learned | Start with rules |

### Datasets (Optional)

| Dataset | Purpose | Size |
|---------|---------|------|
| ScanNet | Indoor scene layouts | Not needed initially |

---

## Module 3: Asset Generator

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| Shap-E | HuggingFace | `pip install diffusers` + 2GB download |
| CUDA GPU | Required | 4GB+ VRAM |

### Datasets

**None needed** - uses pretrained Shap-E

---

## Module 4: Motion Generator

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| MDM model | [GitHub](https://github.com/GuyTevet/motion-diffusion-model) | ~1GB checkpoint |
| HumanML3D dataset | [Link](https://github.com/EricGuo5513/HumanML3D) | Only for training (not inference) |
| SMPL body model | [SMPL](https://smpl.is.tue.mpg.de/) | Registration required |

### Datasets

| Dataset | Purpose | Size | Needed? |
|---------|---------|------|---------|
| HumanML3D | Motion + text pairs | 14K motions | ❌ Inference only |
| AMASS | Raw mocap | 40+ hours | ❌ Not initially |

### Paper References

- MDM: [Tevet et al., ICLR 2023](https://arxiv.org/abs/2209.14916)
- SMPL: [Loper et al., SIGGRAPH 2015](https://smpl.is.tue.mpg.de/)

---

## Module 5: Physics Engine

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| PyBullet | pip | `pip install pybullet` |
| Humanoid URDF | PyBullet data | Included with pybullet |

### Datasets

**None needed** - simulation engine

---

## Module 6: RL Controller (Optional)

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| stable-baselines3 | pip | `pip install stable-baselines3` |
| PyBullet envs | pip | `pip install pybullet` |

### Datasets

**None needed** - learns from simulation

### Training Time

- ~4 hours for basic walking on GPU

---

## Module 7: Render Engine

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| OpenCV | pip | `pip install opencv-python` |
| imageio | pip | For video export |

### Datasets

**None needed** - rendering code

---

## Module 8: AI Enhancer

### Prerequisites

| Requirement | Source | Notes |
|-------------|--------|-------|
| ControlNet | HuggingFace | ~5GB download |
| Stable Diffusion | HuggingFace | Included with ControlNet |
| CUDA GPU | Required | 6GB+ VRAM recommended |

### Datasets

**None needed** - pretrained models

---

## Quick Start Order

```
1. Create shared vocabulary (vocabulary.py)
2. Set up Module 1 (Prompt Parser) - depends on vocabulary
3. Set up Module 5 (Physics Engine) - can run standalone
4. Set up Module 7 (Render Engine) - visualize physics
5. Integrate 1 → 5 → 7 (basic pipeline)
6. Add Module 3 (Asset Generator) - 3D objects
7. Add Module 4 (Motion Generator) - humanoid motion
8. Add Module 2 (Scene Planner) - better layouts
9. Add Module 8 (AI Enhancer) - visual quality
10. Add Module 6 (RL Controller) - robustness (optional)
```

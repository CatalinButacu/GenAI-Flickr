# Training and Finetuning Requirements

This document outlines which components require training, available datasets, and recommended approaches.

---

## Overview: Training Requirements

| Component | Needs Training? | Pretrained Available? | Dataset |
|-----------|-----------------|----------------------|---------|
| M1: Prompt Parser | ❌ No (rule-based) | N/A | - |
| M2: Scene Planner | ❌ No (heuristics) | N/A | - |
| M3: Asset Generator | ✓ Optional | ✅ Shap-E | ShapeNet |
| **M4: Motion Generator** | ✅ **Yes** | ✅ MDM/MLD | **HumanML3D** |
| M5: Physics Engine | ❌ No (simulation) | N/A | - |
| **M6: RL Controller** | ✅ **Yes** | Partial | MuJoCo Humanoid |
| M7: Render Engine | ❌ No (PyBullet) | N/A | - |
| M8: AI Enhancer | ✓ Optional | ✅ ControlNet | LAION |
| **SSM Layers** | ✅ **Yes (our contrib)** | ❌ No | **HumanML3D** |

---

## 1. Motion Generator (M4) - HIGH PRIORITY

### What Needs Training

The motion generator converts text → joint angles. Current implementation uses:

- **Placeholder**: Sine waves (no training, low quality)
- **MDM**: Pretrained available (high quality)
- **SSM (ours)**: Needs training (novel contribution)

### Datasets

#### HumanML3D (Recommended)

- **Paper**: <https://arxiv.org/abs/2205.01061>
- **Size**: 14,616 motions, 44,970 text descriptions
- **Format**: SMPL joint angles + text pairs
- **Download**: <https://github.com/EricGuo5513/HumanML3D>

```
data/HumanML3D/
├── new_joint_vecs/     # Joint angle sequences
├── new_joints/         # 3D joint positions  
├── texts/              # Text descriptions
└── Mean.npy, Std.npy   # Normalization stats
```

#### KIT-ML (Alternative)

- **Paper**: <https://arxiv.org/abs/1609.04733>
- **Size**: 3,911 motions
- **Use**: Smaller, faster experiments

### Training Plan for SSM Motion

```python
# Pseudo-code for training SSM motion generator

from src.ssm import MotionSSM
from src.motion_generator import SSMMotionConfig

# 1. Load HumanML3D
train_data = load_humanml3d("data/HumanML3D")

# 2. Create model
config = SSMMotionConfig(d_model=256, d_state=32, n_layers=4)
model = MotionSSM(**config)

# 3. Training loop
for epoch in range(100):
    for text, motion in train_data:
        # Forward pass
        pred = model(text_embedding)
        
        # Loss: MSE on joint angles
        loss = mse_loss(pred, motion)
        
        # Backward
        loss.backward()
        optimizer.step()
```

### Estimated Training Time

| Hardware | Epochs | Time |
|----------|--------|------|
| RTX 3080 | 100 | ~12 hours |
| RTX 4090 | 100 | ~6 hours |
| CPU only | 100 | ~3 days |

---

## 2. RL Controller (M6) - MEDIUM PRIORITY

### What Needs Training

Reinforcement learning for robust humanoid locomotion:

- Walking on uneven terrain
- Recovery from perturbations
- Object manipulation

### Datasets / Environments

#### MuJoCo Humanoid-v4

- Built into Gymnasium
- Standard benchmark
- 17 actuated joints

```python
import gymnasium as gym
env = gym.make("Humanoid-v4")
```

#### DeepMimic Reference Motions

- **Paper**: <https://arxiv.org/abs/1804.02717>
- **Use**: Motion imitation training
- Example motions: walking, running, backflip

#### CMU Motion Capture

- **URL**: <http://mocap.cs.cmu.edu/>
- **Size**: 2,605 motion clips
- **Use**: Reference for motion imitation

### Training Plan

```python
from stable_baselines3 import PPO
import gymnasium as gym

# 1. Create environment
env = gym.make("Humanoid-v4")

# 2. Train with PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000_000)

# 3. Save
model.save("rl_humanoid_walk")
```

### Estimated Training Time

| Task | Timesteps | Time (RTX 3080) |
|------|-----------|-----------------|
| Basic walk | 1M | 2 hours |
| Robust walk | 10M | 20 hours |
| Complex skills | 50M | 3-5 days |

---

## 3. SSM Layers - YOUR CONTRIBUTION

### What to Train

Your novel **PhysicsSSM** component needs:

- Motion data (HumanML3D)
- Physics constraints (from PyBullet)

### Novel Training Approach

```python
# Physics-constrained SSM training

def train_physics_ssm():
    model = PhysicsSSM(d_model=256, d_physics=64)
    
    for motion, physics_state in data_loader:
        # motion: from HumanML3D
        # physics_state: from PyBullet simulation
        
        # Forward with physics constraints
        pred = model(motion, physics_state)
        
        # Loss 1: Motion reconstruction
        motion_loss = mse(pred, motion)
        
        # Loss 2: Physics consistency (novel!)
        # Check if output violates physics
        physics_loss = compute_physics_violation(pred)
        
        # Combined loss
        loss = motion_loss + 0.1 * physics_loss
```

### This is Novel Because

1. Standard motion models: Learn from data only
2. Physics simulation: No learning
3. **Your PhysicsSSM**: Combines both → learned physics-aware motion

---

## 4. Asset Generator (M3) - LOW PRIORITY

### Pretrained Models Available

| Model | Link | Quality |
|-------|------|---------|
| Shap-E | <https://github.com/openai/shap-e> | Good |
| TripoSR | <https://huggingface.co/stabilityai/TripoSR> | Better |

### Dataset (if finetuning)

- **ShapeNet**: <https://shapenet.org/>
- **Objaverse**: <https://objaverse.allenai.org/>

Finetuning is optional - pretrained models work well.

---

## 5. AI Enhancer (M8) - LOW PRIORITY

### Pretrained Models

- **ControlNet**: <https://github.com/lllyasviel/ControlNet>
- **Stable Diffusion**: Use as base

### Dataset (if finetuning)

- **LAION-5B**: General images
- Custom: Pairs of (physics render, enhanced image)

---

## Recommended Training Order

```
Week 1-2: Download HumanML3D, setup training pipeline
    └── Test with small subset first
    
Week 3-4: Train SSM Motion Generator
    └── Compare with MDM baseline
    
Week 5-6: Train PhysicsSSM (your contribution)
    └── Add physics loss term
    
Week 7-8: (Optional) Train RL Controller
    └── For robust locomotion demo
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8GB | 16GB+ |
| RAM | 16GB | 32GB |
| Storage | 50GB | 200GB |
| Training | RTX 3060 | RTX 4080/4090 |

---

## Quick Start Commands

```bash
# 1. Download HumanML3D
git clone https://github.com/EricGuo5513/HumanML3D
cd HumanML3D && bash prepare/download_glove.sh

# 2. Install dependencies
pip install torch transformers datasets

# 3. Run training (once script is created)
python train_ssm_motion.py --epochs 100 --batch_size 32
```

---

## References

1. **HumanML3D**: Guo et al., CVPR 2022 - <https://arxiv.org/abs/2205.01061>
2. **MDM**: Tevet et al., ICLR 2023 - <https://arxiv.org/abs/2209.14916>
3. **Motion Mamba**: Zhang et al., ECCV 2024 - <https://arxiv.org/abs/2403.07487>
4. **PPO**: Schulman et al., 2017 - <https://arxiv.org/abs/1707.06347>
5. **DeepMimic**: Peng et al., SIGGRAPH 2018 - <https://arxiv.org/abs/1804.02717>

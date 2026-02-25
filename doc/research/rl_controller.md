# RL Controller - Deep Research Analysis

## The Giants We Stand On

### 1. PPO in PyBullet (OpenAI, 2017+)

**Paper**: Schulman et al., "Proximal Policy Optimization Algorithms"
**Link**: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

#### What They Solved

- Stable policy gradient learning
- Works for continuous control (humanoid walking)
- Standard for physics-based character control

#### What We Borrow

| Concept | How We Use It |
|---------|--------------|
| PPO algorithm | Train locomotion policies |
| Humanoid env | PyBullet's `HumanoidBulletEnv` |
| Pretrained | Use pybullet-gym checkpoints |

---

### 2. Soft Actor-Critic (SAC)

**Paper**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"

#### When to Use Instead of PPO

| Scenario | PPO | SAC |
|----------|-----|-----|
| Sample efficiency | Lower | Higher |
| Training stability | Higher | Lower |
| Exploration | Less | More |

---

## Do You NEED RL?

### When RL is REQUIRED

| Scenario | Why |
|----------|-----|
| Uneven terrain | Adapt foot placement |
| External forces | Recover from push |
| Novel situations | Handle what motion data doesn't cover |

### When RL is OVERKILL

| Scenario | Simpler Alternative |
|----------|---------------------|
| Walk forward | Use pretrained MDM |
| Dance sequence | Motion capture |
| Scripted action | Keyframe animation |

---

## YOUR DECISION

### Recommendation for Dissertation
>
> **Skip RL initially. Add as "future work" or final phase.**

### Justification

1. MDM provides good motion quality without training
2. RL training takes 1-3 days per behavior
3. Focus on pipeline integration first
4. RL adds value for edge cases (robustness)

### If You DO Implement RL

**Easiest path**:

1. Use `stable-baselines3` library
2. Load PyBullet Humanoid environment
3. Train PPO for 1M steps (~4 hours on GPU)
4. Use for "recover from push" demos

```python
from stable_baselines3 import PPO
from pybullet_envs import HumanoidBulletEnv

env = HumanoidBulletEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("humanoid_walk")
```

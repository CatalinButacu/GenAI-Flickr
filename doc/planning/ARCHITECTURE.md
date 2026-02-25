# Extended Pipeline Architecture - Deep Design Document

## Your Vision (Translated)
>
> "NLP → Figure Generation → PyBullet Physics → RL for Character Actions → ML for Mesh + Reality"

This document breaks down EVERY module, what needs training, what's pretrained, and how they connect.

---

## Complete Modular Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FULL PIPELINE (8 MODULES)                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │  1. PROMPT  │──▶│  2. SCENE   │──▶│  3. ASSET   │──▶│  4. MOTION  │         │
│  │   PARSER    │   │   PLANNER   │   │  GENERATOR  │   │  GENERATOR  │         │
│  │   (NLP)     │   │   (NEW!)    │   │  (3D Mesh)  │   │  (NEW!)     │         │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘         │
│        │                 │                 │                 │                  │
│        ▼                 ▼                 ▼                 ▼                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐         │
│  │  5. PHYSICS │──▶│  6. RL      │──▶│  7. RENDER  │──▶│  8. ENHANCE │         │
│  │   ENGINE    │   │  CONTROLLER │   │   ENGINE    │   │   (AI)      │         │
│  │  (PyBullet) │   │  (NEW!)     │   │  (Camera)   │   │ (ControlNet)│         │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### Module 1: Prompt Parser (NLP)

| Aspect | Details |
|--------|---------|
| **Input** | Natural language: "A robot walks to a ball and kicks it" |
| **Output** | Structured JSON: actors, objects, actions, scene |
| **Pretrained?** | ✅ Yes - Use spaCy or GPT for parsing |
| **Training needed?** | ❌ No - Rule-based + LLM fallback |
| **Speed** | ⚡ <1s |
| **Implementation** | Extend current `StoryAgent` |

```python
# Output example
{
    "actors": [{"type": "humanoid", "name": "robot"}],
    "objects": [{"type": "sphere", "name": "ball"}],
    "actions": [
        {"actor": "robot", "action": "walk_to", "target": "ball"},
        {"actor": "robot", "action": "kick", "target": "ball"}
    ]
}
```

---

### Module 2: Scene Planner (NEW!)

| Aspect | Details |
|--------|---------|
| **Input** | Parsed JSON from Module 1 |
| **Output** | Scene graph with positions, relationships |
| **Pretrained?** | ✅ Partial - Use layout priors |
| **Training needed?** | ⚠️ Optional - fine-tune for your domain |
| **Speed** | ⚡ <1s |

**Purpose**: Decide WHERE objects go before physics starts.

- Ball should be 2m away from robot
- Robot facing the ball
- Camera positioned for good view

---

### Module 3: Asset Generator (3D Meshes)

| Aspect | Details |
|--------|---------|
| **Input** | Object descriptions: "wooden ball", "blue cube" |
| **Output** | OBJ/GLTF mesh files |
| **Pretrained?** | ✅ Yes - Shap-E, TripoSR |
| **Training needed?** | ❌ No (unless you want custom style) |
| **Speed** | ⏱️ 5-15s per object |

**Performance tip**: Cache generated meshes by prompt hash!

---

### Module 4: Motion Generator (NEW! - Critical)

| Aspect | Details |
|--------|---------|
| **Input** | Action command: "walk", "kick", "jump" |
| **Output** | Motion sequence (joint angles over time) |
| **Pretrained?** | ✅ Yes - MDM, T2M, HuMotion |
| **Training needed?** | ⚠️ Depends on complexity |
| **Speed** | ⏱️ 1-5s per motion |

**Options (ranked by ease)**:

| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| **MDM (Motion Diffusion)** | High quality, text input | Slower | High quality demos |
| **MoConVQ** | Fast motion matching | Less diverse | Real-time |
| **HuMotion 1.0** | State of art | Large model | Production |

**Key Insight**: You DON'T need to train from scratch!

```python
# Example with pretrained MDM
from motion_diffusion_model import MDM
model = MDM.from_pretrained("humanml3d")
motion = model.generate("a person walks forward")  # Returns joint angles!
```

---

### Module 5: Physics Engine (PyBullet)

| Aspect | Details |
|--------|---------|
| **Input** | Meshes + motion targets + scene layout |
| **Output** | Physics-corrected positions per frame |
| **Pretrained?** | ✅ N/A - physics library |
| **Training needed?** | ❌ No |
| **Speed** | ⚡ Real-time (240Hz simulation) |

**Critical Role**:

- Motion Gen gives "ideal" motion
- Physics makes it REAL (gravity, collisions, friction)

---

### Module 6: RL Controller (NEW! - Optional but powerful)

| Aspect | Details |
|--------|---------|
| **Input** | Current character state + goal |
| **Output** | Joint torques to achieve goal |
| **Pretrained?** | ⚠️ Partially - PPO/SAC agents exist |
| **Training needed?** | ⚠️ Yes for custom behaviors |
| **Speed** | ⚡ <1ms per step (inference) |

**When to use RL vs Motion Gen**:

| Scenario | Use Motion Gen | Use RL |
|----------|---------------|--------|
| "walk forward" | ✅ | ❌ |
| "walk on uneven terrain" | ❌ | ✅ |
| "recover from push" | ❌ | ✅ |
| "dance salsa" | ✅ | ❌ |

**Recommendation**: Start WITHOUT RL. Add later for robustness.

---

### Module 7: Render Engine (Camera)

| Aspect | Details |
|--------|---------|
| **Input** | Physics state at each frame |
| **Output** | RGB + Depth images |
| **Pretrained?** | ✅ N/A - rendering code |
| **Training needed?** | ❌ No |
| **Speed** | ⚡ 10-30 FPS |

Already implemented in your `Simulator.render()`.

---

### Module 8: AI Enhancement (ControlNet)

| Aspect | Details |
|--------|---------|
| **Input** | Depth frames + style prompt |
| **Output** | Photorealistic frames |
| **Pretrained?** | ✅ Yes - ControlNet Depth |
| **Training needed?** | ❌ No (fine-tune for style) |
| **Speed** | ⏱️ 0.5-1s per frame |

**Bottleneck!** This is your slowest step.

---

## Training Requirements Summary

| Module | Train? | Data Needed | Time to Train | Alternative |
|--------|--------|-------------|---------------|-------------|
| Prompt Parser | ❌ | - | - | GPT/spaCy |
| Scene Planner | ⚠️ | Scene layouts | 1-2 days | Heuristics |
| Asset Generator | ❌ | - | - | Shap-E pretrained |
| Motion Generator | ⚠️ | Motion capture | 1 week | MDM pretrained |
| Physics Engine | ❌ | - | - | PyBullet |
| RL Controller | ⚠️ | Rewards only | 1-3 days | Skip initially |
| Render Engine | ❌ | - | - | Code |
| AI Enhancement | ❌ | - | - | ControlNet pretrained |

---

## Performance Analysis (30s video @ 24fps = 720 frames)

### Without RL (Simpler path)

| Step | Per-item Time | Total Time | GPU Memory |
|------|--------------|------------|------------|
| Parse | 0.5s | 0.5s | 0 |
| Asset Gen (3 objects) | 15s | 45s | 4GB |
| Motion Gen (1 action) | 3s | 3s | 2GB |
| Physics (720 frames) | 0.003s | 2.2s | 0 |
| Render | 0.03s | 22s | 2GB |
| ControlNet (720 frames) | 0.7s | **504s (~8min)** | 5GB |
| **Total** | | **~9.5 minutes** | |

### Speed Optimizations

1. **Skip ControlNet for dev**: Raw PyBullet video = 2 minutes total
2. **Keyframe ControlNet**: Enhance every 12th frame → 42 frames = 30s
3. **Lower res first**: 256x256 instead of 512x512 = 4x faster
4. **Cache assets**: Don't regenerate same objects

---

## Recommended Implementation Phases

### Phase 1: Core Pipeline (2 weeks)

- [x] Prompt Parser (existing)
- [ ] Object Asset Gen (existing, optimize)
- [ ] Physics Engine (existing)
- [ ] Render (existing)
**Deliverable**: Basic object animation from text

### Phase 2: Character Motion (3 weeks)

- [ ] Integrate MDM for humanoid motion
- [ ] SMPL body model in PyBullet
- [ ] Motion → Physics retargeting
**Deliverable**: "person walks" → video

### Phase 3: Intelligence (2 weeks)

- [ ] Scene Planner module
- [ ] RL for physical interaction
**Deliverable**: "robot picks up box" → video

### Phase 4: Polish (2 weeks)

- [ ] ControlNet temporal consistency
- [ ] Performance optimization
- [ ] Demo videos for dissertation
**Deliverable**: Final quality demos

---

## File Structure Proposal

```
src/
├── prompt_parser/           # Module 1 (was story_agent)
│   ├── __init__.py
│   ├── parser.py           # Main NLP logic
│   └── schemas.py          # Output data structures
│
├── scene_planner/           # Module 2 (NEW)
│   ├── __init__.py
│   ├── planner.py          # Layout algorithm
│   └── spatial.py          # Position helpers
│
├── asset_generator/         # Module 3 (was model_generator)
│   ├── __init__.py
│   ├── generator.py        # Shap-E/TripoSR wrapper
│   ├── cache.py            # Mesh caching
│   └── physics_props.py    # Mass/friction estimation
│
├── motion_generator/        # Module 4 (NEW)
│   ├── __init__.py
│   ├── mdm_wrapper.py      # Motion Diffusion Model
│   ├── smpl_adapter.py     # SMPL body handling
│   └── retarget.py         # Motion → Physics joints
│
├── physics_engine/          # Module 5 (existing)
│   ├── __init__.py
│   ├── scene.py
│   ├── simulator.py
│   └── characters.py       # Humanoid physics body (NEW)
│
├── rl_controller/           # Module 6 (NEW, optional)
│   ├── __init__.py
│   ├── agent.py            # PPO/SAC agent
│   ├── rewards.py          # Reward functions
│   └── pretrained/         # Saved models
│
├── render_engine/           # Module 7 (was in simulator)
│   ├── __init__.py
│   ├── camera.py           # Cinematic camera
│   └── video.py            # Video export
│
├── ai_enhancer/             # Module 8 (was video_renderer)
│   ├── __init__.py
│   ├── controlnet.py       # ControlNet wrapper
│   ├── temporal.py         # Temporal consistency
│   └── styles.py           # Style presets
│
└── pipeline.py              # Main orchestrator
```

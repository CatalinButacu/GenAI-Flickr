# Motion Generator Module

Text-to-motion generation with multiple backends.

## Backends

| Backend       | Source          | Quality             | Speed  | Use Case   |
|---------------|-----------------|---------------------|--------|------------|
| **retrieval** | KIT-ML dataset  | ⭐⭐⭐ Ground truth | Fast   | Production |
| **ssm**       | Trained SSM     | ⭐⭐ Learning       | Medium | Research   |
| **placeholder** | Procedural    | ⭐ Basic            | Fast   | Fallback   |

## Architecture

```text
Text: "A person walks forward"
           │
           ▼
    ┌──────────────────┐
    │ MotionGenerator  │
    └────────┬─────────┘
             │
    ┌────────┼────────────────┐
    │        │                │
    ▼        ▼                ▼
Retrieval   SSM Model    Placeholder
(KIT-ML)   (Trained)    (Fallback)
    │        │                │
    └────────┼────────────────┘
             │
             ▼
      MotionClip(frames, fps, source)
```

## Usage

```python
from src.motion_generator import MotionGenerator

gen = MotionGenerator()

# Get real motion from dataset (recommended)
clip = gen.generate("A person walks forward", prefer="retrieval")
print(f"Source: {clip.source}, Frames: {clip.num_frames}")

# Use trained SSM model
clip = gen.generate("A person runs", prefer="ssm")
```

## Output Format

**MotionClip:**

- `frames`: `np.ndarray (num_frames, 251)` - KIT-ML format
- `fps`: `int` (default 20)
- `source`: `str` ("retrieved", "generated", "placeholder")
- `duration`: `float` (seconds)

**Motion Dimensions (251):**

- `[0:3]` Root velocity (x, y, z)
- `[3:66]` Joint positions (21 joints × 3)
- `[66:129]` Joint velocities (21 joints × 3)
- `[129:251]` Joint rotations (21 joints × 6D)

## Files

| File          | Purpose                          |
|---------------|----------------------------------|
| `generator.py`    | Main generator with 3 backends |
| `train.py`        | SSM training pipeline          |
| `README.md`       | This documentation             |

## SSM Model Details

When using `prefer="ssm"`:

- **Architecture**: TextEncoder → MambaLayer ×4 → MotionDecoder
- **Parameters**: 5.1M
- **Training**: 20 epochs on KIT-ML
- **Val Loss**: 0.3768

## References

- [Motion Mamba (ECCV 2024)](https://arxiv.org/abs/2403.07487)
- [Mamba (Gu & Dao, 2024)](https://arxiv.org/abs/2312.00752)
- [KIT-ML Dataset](https://motion-annotation.humanoids.kit.edu/)

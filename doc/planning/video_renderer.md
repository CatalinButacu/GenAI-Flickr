# Video Renderer - Technical Planning

## Module Input/Output

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `List[FrameData]` | Frames from Physics Engine (contains depth maps) |
| **Output** | `List[EnhancedFrame]` + MP4 video | Realistic frames + comparison video |

---

## Data Classes

### `EnhancedFrame` (lines 12-18)
| Field | Type | Purpose |
|-------|------|---------|
| `original_rgb` | np.ndarray | PyBullet render |
| `enhanced_rgb` | np.ndarray | ControlNet output |
| `depth` | np.ndarray | Depth conditioning input |
| `timestamp` | float | Frame time |

---

## Class: `VideoRenderer`

### `__init__(self, model_id, controlnet_id, device)`
Sets up model identifiers and device.

### `setup(self) -> bool`
Loads ControlNet + Stable Diffusion pipeline:
1. Load `ControlNetModel` from `lllyasviel/sd-controlnet-depth`
2. Load `StableDiffusionControlNetPipeline`
3. Apply UniPCMultistepScheduler for faster inference
4. Enable xformers if available

### `enhance_frame(self, depth_map, prompt, ...) -> np.ndarray`
Enhances single frame:
1. Convert depth to RGB (stack 3 channels)
2. Run ControlNet pipeline with prompt
3. Return enhanced image

### `enhance_frames(self, frames, prompt, ...) -> List[EnhancedFrame]`
Batch enhancement:
1. Loop through frames
2. Call `enhance_frame()` for each
3. Log progress every 10 frames

### `create_video(self, frames, output_path, fps, comparison) -> str`
Creates MP4:
- `comparison=True`: Side-by-side original|enhanced
- `comparison=False`: Enhanced only

### `create_enhanced_only_video(self, frames, output_path, fps) -> str`
Wrapper for enhanced-only video.

---

## Flow Diagram

```
List[FrameData] ─→ enhance_frames() ─→ for each frame:
       │                                      │
       │                           depth_map + prompt
       │                                      ↓
       │                            enhance_frame() → ControlNet
       │                                      ↓
       │                              EnhancedFrame
       │
       └───────────────────→ create_video() → output.mp4
```

---

## Performance Notes
- ~0.5-1s per frame at 512x512 with 15 inference steps
- 120 frames (5s @ 24fps) ≈ 60-120s enhancement time
- GPU memory: ~5GB with xformers

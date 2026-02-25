# Model Generator - Technical Planning

## Module Input/Output

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `str` (text) or image path | Prompt or reference image |
| **Output** | `GeneratedModel` | Contains `mesh_path` to .obj/.ply file |

---

## Data Classes

### `GeneratedModel` (lines 12-18)
| Field | Type | Purpose |
|-------|------|---------|
| `name` | str | Object name |
| `mesh_path` | str | Path to generated mesh file |
| `preview_path` | Optional[str] | Reference image (if hybrid) |
| `backend` | str | "shap-e" or "triposr" |

---

## Class: `ModelGenerator`

### `__init__(self, backend: Literal["shap-e", "triposr"] = "shap-e", device: str = "cuda")`
Selects generation method. Default is Shap-E for text-to-3D.

### `setup(self) -> bool`
Loads model weights from HuggingFace. First run downloads ~2GB.

### `_setup_shap_e(self) -> bool`
Loads `ShapEPipeline` from `openai/shap-e`.

### `_setup_triposr(self) -> bool`
Loads `TSR` model from StabilityAI.

### `generate_from_text(self, prompt: str, output_dir: str, ...) -> GeneratedModel`
Main text-to-3D function:
1. Run diffusion with `guidance_scale` and `num_inference_steps`
2. Export mesh to PLY and OBJ
3. Return `GeneratedModel` with paths

### `generate_from_image(self, image_path: str, output_dir: str, ...) -> GeneratedModel`
Image-to-3D via TripoSR:
1. Load image
2. Run reconstruction
3. Export OBJ mesh

### `generate_preview_image(self, prompt: str, ...) -> str`
Optional: Uses Stable Diffusion to create reference image for hybrid pipeline.

---

## Flow Diagram

```
Text Prompt ─→ generate_from_text() ─→ Shap-E Pipeline ─→ .obj/.ply
                                                              ↓
                                                    GeneratedModel

Image Path ─→ generate_from_image() ─→ TripoSR ─→ .obj
```

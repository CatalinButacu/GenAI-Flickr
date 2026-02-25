# Story Agent - Technical Planning

## Module Input/Output

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `str` | Natural language prompt (e.g., "a red ball falls on a table") |
| **Output** | `SceneDescription` | Structured scene with objects, actions, camera motions |

---

## Data Classes

### `SceneObject` (lines 10-20)
| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `name` | str | required | Object identifier |
| `shape` | str | "box" | Primitive type or "mesh" |
| `size` | List[float] | [0.1, 0.1, 0.1] | Dimensions |
| `position` | List[float] | [0, 0, 0.5] | Initial XYZ |
| `color` | List[float] | [0.5, 0.5, 0.5, 1.0] | RGBA |
| `mass` | float | 1.0 | kg |
| `is_static` | bool | False | Fixed in place? |
| `mesh_prompt` | Optional[str] | None | Text prompt for 3D generation |

### `SceneAction` (lines 23-29)
Applies forces/velocities to objects at specified times.

### `CameraMotion` (lines 32-39)
Defines orbit, zoom, pitch, pan effects.

### `SceneDescription` (lines 42-51)
Container for full parsed scene.

---

## Class: `StoryAgent`

### `__init__(self, use_llm: bool = False, llm_model: str = "gpt-3.5-turbo")`
Initializes parser mode (rule-based or LLM).

### `setup(self) -> bool`
Loads LLM client if `use_llm=True`. Returns True on success.

### `parse(self, prompt: str) -> SceneDescription`
Main entry point. Delegates to `_parse_rules()` or `_parse_with_llm()`.

### `_parse_rules(self, prompt: str) -> SceneDescription`
Rule-based parsing:
1. Tokenize prompt, search for shape keywords (ball, sphere, cube, box, etc.)
2. Extract colors via regex
3. Detect actions ("falls", "bounces", "throws")
4. Generate default camera orbit

### `_parse_with_llm(self, prompt: str) -> SceneDescription`
Calls OpenAI API with structured JSON prompt. Parses response into dataclasses.

---

## Flow Diagram

```
User Prompt ─→ parse() ─→ [LLM?] ─→ _parse_with_llm()
                   │                     │
                   └── no ──→ _parse_rules()
                                        │
                               SceneDescription
```

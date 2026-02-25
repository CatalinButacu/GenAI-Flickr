# Physics Engine - Technical Planning

## Module Input/Output

| Direction | Type | Description |
|-----------|------|-------------|
| **Input** | `SceneDescription` | Objects and actions from Story Agent |
| **Output** | `List[FrameData]` | RGB/Depth/Segmentation frames |

---

## Data Classes

### `PhysicsObject` (scene.py, lines 11-20)
| Field | Type | Purpose |
|-------|------|---------|
| `name` | str | Identifier |
| `body_id` | int | PyBullet body handle |
| `mass` | float | Object mass in kg |
| `position` | List[float] | Current XYZ |
| `is_static` | bool | Immovable if True |

### `CameraConfig` (simulator.py, lines 14-25)
Camera position/orientation config.

### `FrameData` (simulator.py, lines 28-34)
| Field | Type | Purpose |
|-------|------|---------|
| `timestamp` | float | Simulation time |
| `rgb` | np.ndarray | Color image |
| `depth` | np.ndarray | Depth buffer |
| `segmentation` | np.ndarray | Object ID mask |

### `CinematicCamera` (simulator.py, lines 37-122)
Handles orbit, zoom, pitch, pan with easing.

---

## Class: `Scene` (scene.py)

### `setup(self, use_gui: bool = False) -> bool`
Initializes PyBullet server. Sets gravity.

### `add_ground(self) -> int`
Loads `plane.urdf` ground.

### `add_primitive(self, name, shape, size, mass, position, color, is_static) -> PhysicsObject`
Creates box/sphere/cylinder collision+visual shapes.

### `load_mesh(self, name, mesh_path, mass, position, scale, is_static) -> PhysicsObject`
Loads OBJ mesh into simulation.

### `apply_force(self, name, force, position) -> bool`
Applies external force to object.

### `set_velocity(self, name, linear, angular)`
Sets initial velocity.

### `cleanup(self)`
Disconnects PyBullet.

---

## Class: `Simulator` (simulator.py)

### `step(self, dt)`
Advances physics by `dt` seconds.

### `render(self) -> FrameData`
Captures RGB/Depth/Seg from camera.

### `run(self, duration, fps, actions) -> List[FrameData]`
Basic simulation loop without camera motion.

### `run_cinematic(self, duration, fps, actions, cinematic_camera) -> List[FrameData]`
Main simulation with camera effects:
1. For each frame: update camera, step physics, render
2. Apply actions at specified times

### `create_video(self, frames, output_path, fps, layout)`
Exports frames as MP4 with horizontal/vertical layout.

---

## Flow Diagram

```
SceneDescription ─→ Scene.setup() ─→ add_primitive/load_mesh for each object
                                            │
                              Simulator.run_cinematic()
                                            │
                       ┌─── step physics ──→ render() ──→ FrameData
                       │
                 create_video() ─→ output.mp4
```

# Data Collection

## 1. Collection pipeline overview

`DataCollector` runs inside the PyBullet sim loop (`sim_node.py` calls
`step_collection()` once per physics step, which runs at **240 Hz**). State is
read synchronously on the sim thread. Each render job bundles a state snapshot with the camera matrices needed for that frame, so state and images stay paired even though rendering is async.

- **Capture rate:** configurable `hz` (default 5 Hz), implemented as a frame
  decimation of the 240 Hz physics loop (`capture_frequency = 240 / hz`). Only
  one sample in every `capture_frequency` physics steps is captured.
- **Renderer:** auto-selected — `ER_BULLET_HARDWARE_OPENGL` if a `DISPLAY` is
  present, otherwise `ER_TINY_RENDERER` (software, headless).
- **Success gating:** an episode is only written to disk if `self.success` is
  set `True` before `save()` is called; failed runs are discarded entirely
  (no partial/failure episodes exist in the dataset).

## 2. What is recorded per timestep

For every captured frame, the collector reads:

| Field | Source | Shape (per step) | Notes |
|---|---|---|---|
| `timestamps` | `sim_steps / 240.0` | scalar | Simulation time in seconds, not wall-clock time |
| `proprio` | `p.getJointStates` over **all** robot joints | `(num_joints,)` | Raw joint *positions* only (no velocities/torques captured), `float32` |
| `ee_poses` | End-effector link state | `(7,)` = `[x, y, z, roll, pitch, yaw, gripper_state]` | Absolute pose, Euler angles, world frame |
| `ee_deltas` | Frame-to-frame difference of `ee_poses` | `(7,)` | See **rotation wraparound caveat** below |
| `obj_poses` | `p.getBasePositionAndOrientation` per object in `object_ids` | `(7 * num_objects,)` | Position (3) + **quaternion** (4) per object, concatenated in `object_ids` order |
| images | `p.getCameraImage` per active camera key | `(H, W, 3)` uint8 RGB | One stream per camera in `camera_keys` |

### Proprioception (`proprio`)
Joint order matches `joint_names.txt` (one name per line, same order as the
`(num_joints,)` axis of `proprio.npy`). This includes all robot joints, not
just the arm/hand used for manipulation — if the action space used for
training only covers a subset (e.g. right arm + hand), you'll need to slice
`proprio` accordingly using `joint_names.txt`, not assume a fixed index range.

### End-effector pose (`ee_poses`, `ee_deltas`)
Column order is documented in `ee_columns.txt`: `x,y,z,roll,pitch,yaw,gripper_state`.

- Position is in meters, world frame, taken straight from `getLinkState`
  (no smoothing/filtering).
- Orientation is **Euler angles (roll, pitch, yaw)**, not a quaternion or
  6D rotation representation. 
- `gripper_state` is a **scalar in [0, 1]**, computed as the mean of the
  10 finger joints' positions normalized by their own `(lower, upper)` limit
  range (`FINGER_JOINT_NAMES`, right hand only). It is *not* a binary
  open/close flag and not a raw joint angle — it's a derived, normalized
  aperture proxy. 

#### Delta EE (`ee_deltas`)
Computed as `ee_abs[t] - ee_abs[t-1]`, with rotation components wrapped into
`(-π, π]` to avoid large jumps at the ±π boundary. The **first frame's delta is
always zero** (`_prev_ee_pose is None` case).


### Object poses (`obj_poses`)
Flattened as `[x, y, z, qx, qy, qz, qw] × num_objects`, in the order of
`self.object_ids` at construction time. There is no per-object name or ID
saved alongside this array on disk — the mapping from index to object identity
only exists implicitly via the order `object_ids` was passed into
`DataCollector.__init__`, so this must be tracked/reproduced externally (e.g.
in the calling script) to correctly label objects during RLDS conversion. If
an object pose read fails, it's silently zero-filled (`[0,0,0,0,0,0,0]`)
rather than flagged, so all-zero object poses in the data aren't necessarily
real positions.

### Images
- Resolution defaults to **320×240** (`render_width` / `render_height`),
  RGB only, `uint8`, alpha channel dropped (`[:, :, :3]`). No depth or
  segmentation data is captured.
- `camera_keys` controls which streams exist; typically `r_eye`, `l_eye`
  (head-mounted, robot-pose-dependent) and/or `fixed_camera` (static
  table-mounted view). Unknown camera keys are silently skipped.
- Eye cameras are recomputed every captured frame from the current head/eye
  link pose (`get_pybullet_camera_matrices`) — FOV 60°, aspect 1.33, near
  0.02 / far 10.0. The `fixed_camera` view/projection matrices are computed
  once at `start()` and reused for the whole episode (true static camera),
  using pose loaded from `client.config.camera.fixed_camera_*` or overridden
  by the visualizer's `eye_windows["fixed_camera"]` if present.
- An optional synthetic ground-shadow overlay (`add_ground_shadow`) is
  composited per object per frame based on its projected base position — this
  is a rendering embellishment, not ground-truth geometry, and adds a faint
  dark ellipse under each tracked object.

## 3. On-disk layout (per episode directory)

```
<save_dir>/
├── timestamps.npy        # (N,) float, simulation time in seconds
├── proprio.npy           # (N, num_joints) float32
├── ee_poses.npy          # (N, 7) float32  [x,y,z,roll,pitch,yaw,gripper]
├── ee_deltas.npy         # (N, 7) float32  (frame-to-frame deltas; first row = 0)
├── obj_poses.npy         # (N, 7*num_objects) float32
├── joint_names.txt       # one joint name per line, matches proprio.npy column order
├── ee_columns.txt        # "x,y,z,roll,pitch,yaw,gripper_state"
├── camera_info.json      # fixed camera pose: position/target/up vectors
└── <camera_key>_video.npz   # one per camera, key "video" -> (N, H, W, 3) uint8
```

`N` is the number of captured (decimated) frames, i.e. episode length at the
collection `hz`, not the number of physics steps.

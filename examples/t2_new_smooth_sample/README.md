# T2 smooth sample episode

This directory contains a four-observation excerpt from a real successful
`t2_new_smooth` demonstration. It is small enough to keep in Git and shows the
pipeline's post-processed array and camera schemas; it is not a training set.

Load it with NumPy:

```python
import numpy as np

with np.load("examples/t2_new_smooth_sample/sample_episode.npz") as sample:
    print(sample.files)
    print(sample["proprio"].shape)       # (4, 75)
    print(sample["ee_deltas"].shape)     # (3, 7)
    print(sample["fixed_camera"].shape)  # (4, 240, 320, 3)
```

The archive contains:

| Key | Shape | Meaning |
| --- | --- | --- |
| `timestamps` | `(4,)` | Simulation timestamps for observations 40--43 |
| `proprio` | `(4, 75)` | Robot joint positions |
| `ee_poses` | `(4, 7)` | End-effector position, Euler orientation, and measured gripper state |
| `ee_deltas` | `(3, 7)` | Actions aligned as observation `t` to pose `t+1` |
| `obj_poses` | `(4, 14)` | Bottle and computer position/quaternion poses |
| `fixed_camera` | `(4, 240, 320, 3)` | Fixed-camera RGB frames |
| `l_eye` | `(4, 240, 320, 3)` | Left-eye RGB frames |
| `r_eye` | `(4, 240, 320, 3)` | Right-eye RGB frames |

For `ee_deltas`, columns 0--2 are world-frame translation deltas, columns
3--5 are XYZ Euler rotation deltas, and column 6 is the post-processed absolute
binary gripper command (`0` closed, `1` open). See `metadata.json` for provenance
and alignment details. The original machine-specific source path is omitted.

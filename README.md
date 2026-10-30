# iCub Dataset Generation

Simulation pipeline for generating humanoid manipulation demonstrations with the
simulated iCub robot using pyCub and PyBullet.

This repository was developed by Rita Alexandra da Costa Mota for the master's
thesis *Application of VLA Models for Manipulation in Humanoid Robots* in
Electrical and Computer Engineering at Instituto Superior Tecnico (IST),
Portugal. The pipeline supports research on adapting Vision-Language-Action
(VLA) models to humanoid manipulation.

The generated datasets are available in the
[iCub VLA Manipulation Dataset collection](https://huggingface.co/collections/ritamota/icub-vla-manipulation-dataset).

## Scope

The public workflow covers the five smooth task variants used for dataset
generation:

| CLI task | Demonstration |
| --- | --- |
| `t1_smooth` | Stack the red cube on the blue cube. |
| `t2_new_smooth` | Move the bottle in front of the laptop. |
| `t3_smooth` | Push the green box to the center of the table. |
| `t4_smooth` | Remove the lid from the box. |
| `t5_smooth` | Clean a simulated spill with a sponge. |

Older task implementations remain in the source tree as research history, but
they are not part of the supported public interface.

## Repository layout

```text
configs/       pyCub scene and simulator configurations
core/          data collection, robot skills, and spill generation
tasks/         task definitions and smooth expert controllers
tools/         offline rendering and frame extraction
scripts/       dataset inspection and conversion utilities
recordings/    prompt pools; generated sessions are ignored by Git
tests/         lightweight behavioral contract tests
baseline/      recorded pre-refactoring environment and validation results
legacy/        unsupported historical snapshots retained for reference
docs/          implementation and gripper-controller details
```

## Requirements

The recorded development environment used:

- Ubuntu under WSL2;
- Python 3.12.3;
- PyBullet at 240 physics steps per second;
- the project-specific
  [pyCub fork](https://github.com/rita-mota/pyCub-icub-vla), pinned in
  `requirements.txt`.

A Linux environment is recommended. The interactive Open3D visualizer requires
a working display/OpenGL setup. Offline rendering can use `xvfb-run`.

The pyCub dependency is pinned to commit
`ec02eea7835f28879327726e427f471386720193`. That commit must be present in
the public pyCub fork before installing this repository from GitHub.

## Installation

```bash
git clone https://github.com/rita-mota/icub-sim-data-generation.git
cd icub-sim-data-generation

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For plotting and video helper scripts:

```bash
python -m pip install -r requirements-tools.txt
```

On Debian or Ubuntu, the offline renderer also expects `xvfb-run`:

```bash
sudo apt-get install xvfb
```

TensorFlow/RLDS, LeRobot, and Hugging Face upload utilities are optional
post-processing workflows and are not installed by the default requirements.
They should be used in a separate environment appropriate for the target
dataset-tool versions.

## Generate demonstrations

Run commands from the repository root. For a visual smoke test that does not
save data:

```bash
python main_collect.py --task t1_smooth --episodes 1 --debug
```

Debug mode keeps the simulator window open after the task. Press `Ctrl+C` to
exit.

To collect successful demonstrations:

```bash
python main_collect.py --task t1_smooth --episodes 10
```

Replace `t1_smooth` with any supported task listed above. The runner retries
failed attempts until the requested number of successful episodes is reached.
Only successful episodes are saved.

Object/camera randomization is disabled by default. The existing research
runner accepts `--randomizer True` to enable each task's configured
randomizer. A repository-wide seed is not currently exposed, so exact
randomized episode reproduction is a known limitation.

## Output format

Collection creates:

```text
recordings/<task>/<task>_session_<timestamp>/
  episode_001/
    timestamps.npy
    proprio.npy
    ee_poses.npy
    ee_deltas.npy
    obj_poses.npy
    joint_names.txt
    ee_columns.txt
    camera_info.json
    metadata.json
```

For `N` captured steps, `J` robot joints, and `K` task objects:

| File | Shape/content |
| --- | --- |
| `timestamps.npy` | `(N,)` simulation timestamps |
| `proprio.npy` | `(N, J)` joint positions |
| `ee_poses.npy` | `(N, 7)`: position, Euler orientation, gripper state |
| `ee_deltas.npy` | Raw: `(N, 7)` previous-to-current deltas. After next-action conversion: `(N-1, 7)` current-to-next labels. |
| `obj_poses.npy` | `(N, 7*K)`: position and quaternion per object |
| `camera_info.json` | fixed-camera position, target, and up vector |
| `metadata.json` | task, instruction, objects, config, and gripper profile |

The first six action values are Cartesian and Euler deltas. After gripper
post-processing, the seventh value is the absolute binary gripper command
(`0` closed, `1` open), not a delta.

The collector samples at 5 Hz while physics runs at 240 Hz.

### Small recorded sample

[`examples/t2_new_smooth_sample`](examples/t2_new_smooth_sample) contains four
consecutive observations, three aligned next-action labels, and the matching
frames from all three cameras. It is an excerpt from a real successful
`t2_new_smooth` demonstration and is intended for schema inspection only.
The full datasets are available from the linked Hugging Face collection.

### Camera post-processing

The main collection entry point intentionally records state first and renders
the three camera streams afterward. This keeps camera generation explicit and
separate from task execution.

Render and decode all episodes in a session:

```bash
python process_dataset.py \
  --dataset_dir recordings/t1_smooth/<session_name> \
  --action both
```

This produces `fixed_camera_video.npz`, `l_eye_video.npz`, and
`r_eye_video.npz`, followed by corresponding PNG directories. These generated
files are ignored by Git.

### Dataset diagnostics and Hub upload

Scan a generated session for alignment problems:

```bash
python scripts/scan.py recordings/t1_smooth/<session_name>
```

The Hugging Face uploader is deliberately a dry run unless `--yes` is given:

```bash
python scripts/upload_huggingface.py \
  --folder /path/to/prepared/dataset \
  --repo-id username/dataset-name
```

Review the printed plan, authenticate with `huggingface-cli login`, and add
`--yes` only when the destination and options are correct. The destructive
`--replace` option deletes remote files that are absent locally.

## Validation

Run the lightweight behavioral contracts from the repository root:

```bash
python -m unittest discover -s tests -p "test_baseline_contracts.py" -v
python -m unittest -v tests.test_public_sample
```

The suite checks configuration and prompt hashes, task registration,
minimum-jerk references, fixed-seed spill sampling, gripper profiles, and a
small dataset-output fixture. It does not run a full simulator episode.

The original source is preserved by the `pre-publication-original` tag.
Recorded results and unverified areas are documented in
[baseline/RESULTS.md](baseline/RESULTS.md).

## Documentation

- [Pipeline implementation notes](docs/PIPELINE_IMPLEMENTATION.md)
- [Gripper controller](docs/GRIPPER_CONTROLLER.md)
- [Baseline validation](tests/README.md)
- [pyCub asset provenance](https://github.com/rita-mota/pyCub-icub-vla/blob/icub-vla-publication/icub_pybullet/other_meshes/ASSET_SOURCES.md)

## Known limitations

- Full task success rates and contact-heavy simulator determinism have not yet
  been validated across machines.
- Rendering can vary with PyBullet/OpenGL versions and renderer selection.
- Randomized runs do not yet expose a single repository-wide seed.
- Dataset conversion and Hub upload scripts are research utilities and require
  additional review and environment-specific dependencies.
- The code assumes the modified pyCub fork rather than upstream PyPI pyCub.
- Older T2 recordings already published on Hugging Face may contain the stale
  metadata field `base_prompt: "Move the bottle to the left of the laptop."`.
  The executed task, per-episode instructions, prompt pool, and trajectories
  place the bottle in front of the laptop; the source metadata is now corrected.

## License and attribution

The source code in this repository is released under the [MIT License](LICENSE).
pyCub, robot models, and manipulation-object meshes have their own licenses and
attribution requirements; see the pyCub repository and its asset notice.

If you use this pipeline, please cite the repository and the associated thesis.
See [CITATION.cff](CITATION.cff).

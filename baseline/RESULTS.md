# Pre-publication baseline validation results

- Date: 2026-10-30
- Original source reference: `pre-publication-original`
- Original source commit: `624458cd844e5e6fd9c8bde58ec0dae20af1daa8`
- Validation branch: `public-release`
- Execution environment: WSL2 Linux, recorded in `environment.json`

## Lightweight contract suite

Command:

```bash
python -m unittest discover -s tests -p "test_baseline_contracts.py" -v
```

Result: **10 tests passed; 0 failed; 0 skipped.**

Verified contracts:

- Exact hashes of the six original YAML configurations.
- Exact hashes of the five original prompt pools.
- Complete original task registry and the selected public smooth-task subset.
- Robot URDF, right-hand end effector, and 240 Hz configuration invariants.
- Minimum-jerk reference values for all smooth task implementations.
- Repeatable spill generation for seed `12345`.
- Selected gripper profile parameters and command hysteresis.
- Dataset filenames, array shapes, observation/action alignment, first delta,
  camera metadata, and success-gated saving using a two-step temporary fixture.

Numerical comparisons use exact equality where practical. Minimum-jerk and the
selected seeded spill coordinates use absolute tolerance `1e-15` with zero
relative tolerance because they are deterministic NumPy calculations in the
recorded environment.

## Historical root-level gripper tests

Command:

```bash
python -m unittest -v test_gripper_open_refresh test_sticky_gripper
```

Result: **7 tests passed; 1 pre-existing test failed.**

The failing test is
`StickyGripperTest.test_profile_retains_original_success_requirements`.
It expects the `t1_new_smooth` attachment policy to be
`atomic_when_valid`, while the original implementation currently defines
`per_finger_immediate`. Neither the implementation nor the historical test
was changed. The passing baseline contract suite records
`per_finger_immediate` as the observed original behavior.

## Clean pyCub fork compatibility verification

The contract suite was rerun with the clean local pyCub publication commit
`6bc9bdc903271f44da214d3da53145dbe83950d5` installed in the isolated
repository environment.
All **10 tests passed**. The pyCub fork's own lightweight extension suite also
passed **8 tests**, covering joint and object lookup, end-effector pose access,
fixed-camera pose updates, and portable URDF asset references. The committed
public sample suite passed **3 tests**.

## End-to-end simulator smoke validation

A single non-randomized `t2_new_smooth` episode was collected successfully
using Python 3.12.3, PyBullet 3.2.7, NumPy 1.26.4, SciPy 1.17.1, Open3D 0.20.0,
and the clean pyCub commit above.

The normal T2 configuration enables the Open3D GUI. Under WSL's software
Vulkan renderer, GUI rendering dominated every physics step and made the
initial joint interpolation appear stationary. For the bounded smoke test, a
temporary copy of `configs/t2.yaml` changed only `gui.standard` from `True` to
`False`; task parameters, physics, controller rates, trajectories, and dataset
semantics were unchanged.

Results:

- the task reported success on its first attempt;
- 107 aligned observations were saved with shapes `(107, 75)` proprioception,
  `(107, 7)` end-effector poses/deltas, and `(107, 14)` object poses;
- all numerical arrays were finite, timestamps were strictly increasing, and
  raw translation deltas exactly matched consecutive recorded poses;
- offline post-processing produced 107 frames for each of `fixed_camera`,
  `l_eye`, and `r_eye`, all with shape `(107, 240, 320, 3)` and `uint8` dtype;
- the dataset scanner reported zero length mismatches and zero timestamp gaps;
- a representative fixed-camera frame was inspected and showed the robot hand
  holding the bottle above the laptop.

The run emitted joint-limit clipping warnings for small thumb excursions and a
wrist target near its limit. It also emitted a harmless disconnect warning at
interpreter shutdown after the runner had already disconnected explicitly.
Neither warning invalidated the saved episode, but both should remain visible
as known simulator diagnostics.

## Not verified yet

- Repeated success rates, randomized runs, or full executions of the other
  supported smooth tasks.
- Contact-heavy simulator determinism or numerical equivalence across machines.
- Rendered image equivalence across renderers or machines.
- TensorFlow/RLDS, LeRobot, or Hugging Face conversion pipelines.

Passing one T2 smoke episode and the lightweight suites does not claim
equivalence for these unverified areas.

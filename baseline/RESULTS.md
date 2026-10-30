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
`ec02eea7835f28879327726e427f471386720193` first on `PYTHONPATH`.
All **10 tests passed**. The pyCub fork's own lightweight extension suite also
passed **5 tests**, covering joint lookup, fixed-camera pose updates, and
portable URDF asset references.

## Not verified yet

- Full PyBullet task executions or success rates.
- Contact-heavy simulator determinism.
- Rendered image equivalence across renderers or machines.
- End-to-end collection of a real episode using the clean fork.
- TensorFlow/RLDS, LeRobot, or Hugging Face conversion pipelines.

These require the pyCub fork recovery and a separately approved simulator smoke
suite. Passing the lightweight suite does not claim equivalence for these
unverified areas.

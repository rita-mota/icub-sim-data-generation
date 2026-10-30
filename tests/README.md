# Baseline validation

These tests capture lightweight scientific contracts from the
`pre-publication-original` implementation.

Run them from the repository root in an environment containing the required
pyCub fork:

```bash
python -m unittest discover -s tests -p "test_baseline_contracts.py" -v
python -m unittest -v tests.test_public_sample
```

The suite does not launch a simulator episode or generate a dataset. It checks
configuration and prompt hashes, the task registry, minimum-jerk interpolation,
fixed-seed spill generation, gripper profiles and hysteresis, and the small
on-disk dataset schema using temporary data.

The separate public-sample test checks the committed excerpt's shapes,
observation/action alignment, timestamp ordering, and metadata indices.

The two historical root-level gripper tests are intentionally kept separate
because one contains a known pre-existing expectation mismatch recorded in
`baseline/RESULTS.md`.

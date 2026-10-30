# Gripper State Machine

## Smooth T5 cleaning

`tasks/task_5_smooth.py` adds `T5SmoothCleaning` alongside the original T5.
Run a visual test without saving:

```bash
cd /path/to/icub-sim-data-generation
WAYLAND_DISPLAY="" ../envs/pycub_venv/bin/python main_collect.py \
  --task t5_smooth --episodes 1 --debug
```

Remove `--debug` to collect demonstrations under `recordings/t5_smooth/`.
The task uses 60 Hz minimum-jerk Cartesian references with 240 Hz physics.
Approach, descent, pressing, six alternating wiping strips, mop-up, and retreat
are smoothed. Each segment stops smoothly at its endpoint; this is not a
continuous blended wiping path. Cleaning contacts are checked on every physics
step during wiping, and an early-cleaned stroke finishes rather than stopping
abruptly. Speed/acceleration limits are class constants in the task file.

T5 intentionally retains the original open-hand palm attachment, sponge
table-height pinning, spill logging, and success tolerance (at most two
particles remaining). It does not use a binary finger-grasp profile or change
`RobotSkills`. Binary gripper commands alone do not reproduce its palm
attachment/release; that requires separate T5 replay support.

This project controls the humanoid hand with one binary absolute action while
retaining smooth multi-joint finger motion and contact-aware grasping:

- `1` means **open or remain open**.
- `0` means **close or keep holding**.

The policy does not predict individual finger joints. `RobotSkills` converts
the binary command into task-calibrated finger trajectories, detects contacts,
creates grasp constraints, and releases them. The implementation is in
`sim_data_collection/core/robot_skills.py` and is shared by demonstration
collection and inference/replay.

## States and transitions

| Current state | Command | Result |
|---|---:|---|
| `OPEN` | `1` | No change |
| `OPEN` | `0` | Start `CLOSING` |
| `CLOSING` | `0` | Continue closing and contact processing |
| `CLOSING` | `1` | Abort closing, release constraints, start `OPENING` |
| `CLOSING` | `0` + valid stable grasp | Enter `HOLDING` |
| `CLOSING` | `0` + no valid grasp after settling | Enter `CLOSED_EMPTY` |
| `HOLDING` | `0` | Keep holding |
| `HOLDING` | `1` | Release immediately and start `OPENING` |
| `CLOSED_EMPTY` | `0` | No change |
| `CLOSED_EMPTY` | `1` | Start `OPENING` |
| `OPENING` | `1` | Continue opening |
| `OPENING` | `0` | Reverse direction and start `CLOSING` |
| `OPENING` | joints reach open tolerance | Enter `OPEN` |

`CLOSED_EMPTY` is a terminal closed state for the current close command. A new
close attempt requires an intervening open command.

## Command decoding

`_decode_gripper_command()` accepts binary values and tolerates continuous
predictions with hysteresis:

- value `>= 0.7`: open (`1`)
- value `<= 0.3`: close (`0`)
- value in `(0.3, 0.7)`: retain the previous command
- no previous command: default to open

`comm_server/inference.py` currently converts action element 6 separately:
values below `0.9` become close and values at or above `0.9` become open. This
is equivalent for datasets and models that emit strict `0/1` commands.

## Required execution order

Initialize the controller once, after the scene objects exist:

```python
state = skills.initialize_gripper_controller(
    candidate_object_ids=object_ids,
    hand="right",
    profile="t2_new_smooth",
    initial_command=1,
)
```

For every robot action:

```python
command = int(action[6])

skills.execute_gripper_pre_action(command)

# Apply the Cartesian action here.

for _ in range(gripper_settle_steps):
    for _ in range(skills.gripper_physics_substeps):
        sim.update_simulation()
    diagnostics = skills.execute_gripper_post_action(command)
```

The split is intentional:

1. `execute_gripper_pre_action()` handles command edges and starts closing or
   opening without blocking Cartesian control.
2. Physics advances.
3. `execute_gripper_post_action()` advances the finger trajectory, reads new
   contacts, validates the grasp, attaches objects, and completes transitions.

Calling only the pre-action method starts a transition but does not finish it.
The post-action method must run repeatedly at the configured control rate.

## Smooth finger trajectories

Closing and opening targets use quintic minimum-jerk interpolation:

```text
s(u) = 10u^3 - 15u^4 + 6u^5,  0 <= u <= 1
q(u) = q_start + s(u) * (q_target - q_start)
```

This gives zero reference velocity and acceleration at both endpoints. The
default controller runs at 60 Hz with four 240 Hz physics steps per controller
update.

For every controlled finger, joints 2 and 3 are commanded. The open target is
zero clipped to the joint limits. The closed target is:

```text
lower_limit + closure * (upper_limit - lower_limit)
```

`closing_strategy` selects either:

- `simultaneous`: all configured fingers follow one trajectory together.
- `sequential`: fingers close in `closing_order`, each over
  `finger_closing_duration`.

If configured, `thumb_opposition_target` commands the thumb joint 0 before
finger closure. This value is task-specific and should not be treated as a
universal hand constant.

## Contact detection and grasp validation

Only bodies listed in `candidate_object_ids` can be grasped. At each closing
update, the controller selects the candidate touched by the greatest number of
distinct fingers.

`contact_link_policy` controls which hand links count:

- `fingertip`: joint-3 links only.
- `controlled_phalanges`: joint-2 and joint-3 links.

A grasp is valid only when all configured predicates pass:

1. At least `required_fingers` distinct fingers are present.
2. Every `mandatory_fingers` entry is present.
3. At least `minimum_support_fingers` from `support_fingers` are present.
4. The condition remains valid for `contact_confirmation_steps` controller
   updates when required by the attachment policy.

`post_close_settle_duration` is the maximum closed settling period without a
valid contact condition before the controller returns `CLOSED_EMPTY`.

## Contact and attachment behavior

`freeze_on_contact` determines whether a touching finger stops advancing
through its trajectory. A frozen finger is held using `contact_hold_policy`:

- `measured_position`: hold the position measured at first contact.
- `closed_target`: continue requesting the calibrated closed target.

Available attachment policies are:

- `per_finger_immediate`: constrain each actionable finger link at first
  contact. The constrained finger set is used for validation.
- `atomic_when_valid`: finish/validate the grasp, then constrain one contact
  link for every currently contacting finger in one controller update.
- `per_finger_then_validate`: attach fingertips on contact, including contacts
  outside the active sequential phase, but enter `HOLDING` only after the full
  grasp predicate passes. This is the opt-in T1 sticky replay behavior.
- `all_contacts_atomic`: after validation, constrain every currently touching
  hand link. This closely resembles the original task helper but can
  over-constrain PyBullet when finger links continue moving.
- `single_hand_atomic`: validate the physical multi-finger grasp, then create
  one fixed constraint from the rigid end-effector link to the object. T4 uses
  this to avoid inconsistent multi-link constraints while transporting a lid.

All fixed constraints preserve the object pose at attachment time; the object
is not teleported. Starting `OPENING` removes every gripper-owned constraint
before commanding the open trajectory.

## Configured profiles

| Profile | Closing | Required contact | Attachment | Intended use |
|---|---|---|---|---|
| `default` | Simultaneous, 1.5 s | Any 3 fingertips | Per-finger immediate | Generic binary gripper |
| `t2_legacy` | Simultaneous, 1.0 s; opposed thumb | Any 3 fingertips | Per-finger immediate | Existing T2 recordings |
| `t2_new_smooth` | Simultaneous, 1.5 s; opposed thumb | Any 3 fingertips | Per-finger immediate | Smooth T2 collection/replay |
| `t1_new_smooth` | Sequential; thumb, index, middle, ring | Thumb plus 2 support fingers | Atomic per contacting finger | Smooth T1 collection/replay |
| `t1_sticky_fingertips` | Inherits T1 timing | Same T1 predicate | Early fingertip attachment, then validation | Opt-in sensitive T1 replay |
| `t4_new_smooth` | All five simultaneously, 1.5 s | Any 2 controlled phalanges, stable for 0.5 s | Single rigid hand anchor | Smooth T4 lid transport |

Profiles are deliberately task-specific. Change a profile or add a new one
instead of adding task-name conditionals inside the state machine.

## Collection and replay consistency

Smooth tasks set a class-level `GRIPPER_PROFILE`. `BaseTask.save_metadata()`
writes it to each episode as:

```json
{
  "gripper_controller_profile": "t2_new_smooth"
}
```

During ground-truth replay, `--gripper_profile auto` selects this recorded
profile. Episodes without profile metadata use the legacy task mapping and
otherwise fall back to `default`.

The collector's raw gripper value is an achieved hand-closure measurement, not
the command. Before VLA training or binary replay, run:

```bash
cd /path/to/icub-sim-data-generation
python scripts/make_absolute_gripper_actions.py PATH_TO_SESSION --write --plot
```

The converter detects closing/opening motion and writes a latched absolute
command into `ee_deltas.npy`, while preserving the original array as
`ee_deltas_not_absolute_gripper.npy`. The resulting seventh action dimension
remains `1` until closing begins, remains `0` while closing/holding, and returns
to `1` when opening begins.

## Diagnostics

`execute_gripper_post_action()` returns a dictionary containing:

- state before/after, decoded command, and active profile
- closing/opening progress
- current and accumulated contacting fingers
- selected candidate object and stable-contact count
- closing phase/current sequential finger
- attachment and contact-hold policies
- mandatory/support predicate status
- held object ID and constraint IDs/link IDs

Important distinctions:

- `current_contacting_fingers` contains contacts from the current update.
- `contacting_fingers` accumulates fingers observed during the current close.
- `held_object_id` is set only after attachment succeeds and the controller
  enters `HOLDING`.

## Common failure modes

- **The hand never finishes closing:** ensure post-action is called repeatedly
  with physics steps between calls.
- **Immediate `CLOSED_EMPTY`:** inspect candidate IDs, final grasp pose,
  `current_contacting_fingers`, required/mandatory contacts, and the settling
  duration.
- **The hand closes around the wrong body:** restrict `candidate_object_ids` to
  graspable objects.
- **The object follows the hand with a visible gap:** the constraint is active,
  but the physical grasp pose or attachment frame is wrong. Tune the final
  grasp pose, not only the pre-grasp waypoint. Avoid multi-link attachment for
  moving articulated fingers.
- **Replay differs from collection:** verify metadata contains the profile and
  use `--gripper_profile auto`; also verify the seventh action dimension was
  converted to binary absolute commands.
- **Unexpected initial state:** `initial_command` initializes logical state; it
  does not physically move the hand. Physically open the hand first when the
  initial joint configuration is not known.

## Main implementation files

- `sim_data_collection/core/robot_skills.py`: controller, profiles, contacts,
  constraints, and diagnostics.
- `sim_data_collection/tasks/task_1_smooth.py`: T1 collection integration.
- `sim_data_collection/tasks/t2_new_smooth.py`: T2 collection integration.
- `sim_data_collection/tasks/task_4_smooth.py`: T4 collection integration.
- `comm_server/inference.py`: replay/evaluation initialization and action loop.
- `sim_data_collection/scripts/make_absolute_gripper_actions.py`: conversion
  from measured closure to binary absolute actions.

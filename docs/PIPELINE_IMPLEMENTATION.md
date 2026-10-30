# Dataset creation implementation report: smooth experts

**Scope:** `/path/to/icub-sim-data-generation`
**Review date:** 13 September 2026
**Method:** static inspection of the current source and YAML configurations. Collection, rendering, conversion, and training were not executed. This report documents implemented behavior rather than measured performance.

## 1. Purpose and scope

The subsystem generates simulated iCub manipulation demonstrations for learning from visual observations, language, and end-effector actions. A scripted expert controls pyCub/PyBullet, a collector samples achieved robot/object states, and offline utilities reconstruct camera images and prepare training labels.

The expert uses privileged simulator state: object poses, bounding boxes, joint states, and contacts. It does not estimate geometry from images or interpret the language instruction. Language is an accompanying training label rather than an input driving the expert.

| CLI task | Class | Manipulation | Hand implementation |
|---|---|---|---|
| `t1_smooth` | `T1SmoothStacking` | Stack red cube on green cube | Shared `t1_new_smooth` profile |
| `t2_smooth` | `T2SmoothMoving` | Move bottle to the front of laptop | Task-local smooth finger closure |
| `t2_new_smooth` | `T2NewSmoothMoving` | Same bottle-moving trajectory | Shared `t2_new_smooth` profile |
| `t3_smooth` | `T3SmoothPushing` | Push green box to table centre | Open hand; no grasp controller |
| `t4_smooth` | `T4SmoothOpening` | Lift and relocate removable lid | Shared `t4_new_smooth` profile |

Parent task files are discussed only where smooth classes inherit setup, randomization, configuration, or success checks. Their non-smooth expert sequences, T3 simple, T5, and the historical `robot_skills_before_t1_smooth.py` are excluded.

Source: [task registry](#L).

## 2. Architecture

| Layer | Responsibility | Implementation |
|---|---|---|
| Session runner | Simulator initialization, successful-episode target, retries, reset | `main_collect.py` |
| Scene configuration | Robot, initial joints, assets, poses, physics and cameras | `configs/t1.yaml`–`configs/t4.yaml` |
| Task lifecycle | Setup, randomization, expert execution, evaluation, metadata | `tasks/base_task.py` |
| Smooth experts | Geometry, phase sequencing, reference generation, guards | Five smooth task modules |
| Robot skills | Dynamics, transforms, head targeting, contacts and grasp constraints | `core/robot_skills.py` |
| Recording | Step-based sampling, queueing, RAM buffers, persistence | `core/data_collector.py` |
| Visual reconstruction | Restore recorded state and render three cameras | `tools/render_recording.py`, `tools/decoder.py` |
| Dataset preparation | Next-action alignment, rotation and gripper labels, augmentation, export | `scripts/` |

Each smooth class inherits its corresponding task class, which inherits `BaseTask`. `T2NewSmoothMoving` specializes `T2SmoothMoving` by replacing grasp/release routines; Cartesian phases and the success predicate remain shared.

Task and collector operate on the same simulator. The runner wraps `client.update_simulation()` and calls collection whenever it observes an increase in `client.steps_done`. Movement helpers therefore advance collection without calling the collector explicitly.

The reviewed configurations use `r_hand` as end effector and 240 Hz physics. pyCub supplies robot loading, joint handles, Cartesian motion and simulator integration. The smooth modules call this Cartesian interface; they do not implement their own inverse-kinematics solver.

Sources: [runner](#L), [base lifecycle](#L), [collector](#L), [T2 specialization](#L).

## 3. Collection pipeline

### Session initialization

The runner accepts `--task`, `--episodes` (default 10), `--randomizer` (default false), and `--debug`. It selects the class, prepares prompt variations, creates pyCub from the task configuration, and constructs `RobotSkills`.

It saves a pristine PyBullet state and records original body IDs. Recordings go to `recordings/<task>/<timestamped-session>/episode_<attempt>`. Numbering follows attempts, so success-only datasets can contain gaps.

Each attempt creates a collector with `hz=5.0`, tracked task objects, and `camera_keys=[]`. Normal collection therefore records state only. The background worker still transfers queued state snapshots into RAM; camera generation is a separate stage.

### Episode execution

`BaseTask.run()` executes setup → optional randomization → collector start → expert → success check → collector stop/drain → successful data and metadata saving. Failed normal returns are discarded by `DataCollector.save()`.

There is a lifecycle caveat: the update wrapper is installed before setup, and `step_collection()` has no started flag. T3 moves the unused arm and opens the hand during setup. These updates can increment the collection counter and attempt state reads before IDs are initialized. `start()` does not reset that counter. Thus the intended exclusion of preparation is not enforced by a complete collection gate.

### Retry and reset

Collection repeats until the requested successes are reached, with no maximum attempt count. The actual timeout is `240 * 60 * 5 = 72,000` physics steps, or five simulated minutes; the shorter nearby comment is stale. Counting begins before setup.

After an attempt, the runner releases grasp constraints, removes non-pristine bodies, restores the saved state, commands motors to hold restored positions with zero target velocity, and clears joint `set_point`, `start_time`, and `timeout` bookkeeping. T1 setup also removes persistent stacking constraints and restores robot/cube collisions.

The timeout catch marks failure but does not stop the collector in a `finally` block. A timed-out active collector can leave a daemon worker waiting. Other exceptions are not covered by this catch. These are source-level limitations, not observed failures from this review.

Debug mode disables saving and metadata writing, then keeps the simulation open for inspection.

Sources: [runner](#L), [lifecycle](#L), [sampling](#L).

## 4. Smooth trajectory algorithm

### Minimum-jerk interpolation

The experts use the quintic time law

\[
s(u)=10u^3-15u^4+6u^5,\quad 0\leq u\leq1,
\]

with Cartesian position

\[
p(u)=p_0+s(u)(p_1-p_0).
\]

Selected preparation joints use the same law independently. Reference velocity and acceleration are zero at both endpoints. This is smooth timing of a prescribed segment, rather than online optimization of an entire task.

Orientation follows quaternion SLERP between start and target rotations, evaluated at `s(u)`. Translation and rotation share the segment progression. Each segment begins from the achieved current end-effector pose, so the expert does not assume perfect tracking of the previous target.

### Duration selection

For linear distance d and relative rotation angle θ, duration is chosen as

\[
T=\max\left(T_{min},\frac{1.875d}{v_{max}},
\sqrt{\frac{(10/\sqrt3)d}{a_{max}}},
\frac{1.875\theta}{\omega_{max}},
\sqrt{\frac{(10/\sqrt3)\theta}{\alpha_{max}}}\right).
\]

The constants are the maximum first and absolute second derivatives of the unit quintic. Duration therefore accounts for both translational and angular reference limits.

| Experts | Linear speed | Linear acceleration | Angular speed | Angular acceleration |
|---|---:|---:|---:|---:|
| T1, both T2 variants, T4 | 0.08 m/s | 0.15 m/s² | 0.8 rad/s | 1.2 rad/s² |
| T3 | 0.12 m/s | 0.25 m/s² | 1.2 rad/s | 2.0 rad/s² |

T2 lowers limits during approach, leveling, and placement. T3 imposes at least `10/5 = 2` seconds per segment, intended to yield approximately ten recorded samples per major movement.

### Execution rates and limits of smoothness

The executable smooth Cartesian loops issue targets at 60 Hz and advance four 240 Hz physics steps per target. Collection samples every 48 updates, nominally twelve reference updates per recorded sample. T3 docstrings still claim 240 Hz target generation, but its loop implements 60 Hz.

Commands use `wait=False`, `timeout=None`, and `check_collision=False`. Disabling the motion interface's collision check does not disable all PyBullet contacts. Robot self-collisions are separately disabled in the reviewed YAML files.

T1 and T4 repeatedly hold the seven left-arm joints during Cartesian control to limit unused-arm drift. T2 moves the left shoulder away but does not implement this repeated hold.

The limits constrain references, not measured motion under IK errors, dynamics, contact, or constraint forces. A contact-triggered stop can truncate a segment before its zero-velocity endpoint. T1 placement also uses a different incremental helper. Whole-episode minimum-jerk motion and achieved acceleration bounds are therefore not established.

Sources: [T1 trajectories](../tasks/task_1_smooth.py), [T2 trajectories](../tasks/task_2_smooth.py), [T3 trajectories](../tasks/task_3_smooth.py), [T4 trajectories](../tasks/task_4_smooth.py).

## 5. Shared gripper controller

### State machine and execution order

The shared controller uses **1 = open/keep open** and **0 = close/keep holding**. Continuous values ≥0.7 decode to open, values ≤0.3 to close, and intermediate values retain the preceding command, defaulting to open.

| State | Event | Result |
|---|---|---|
| OPEN | Close | CLOSING |
| CLOSING | Validated attachment | HOLDING |
| CLOSING | Completed closure/settling without valid grasp | CLOSED_EMPTY |
| CLOSING, HOLDING, CLOSED_EMPTY | Open | Release constraints; OPENING |
| OPENING | Open target reached | OPEN |
| OPENING | Close | Reverse into CLOSING |
| HOLDING or CLOSED_EMPTY | Continued close | Maintain state |

A failed close requires an intervening open before retry.

`execute_gripper_pre_action()` handles command edges and starts transitions. After physics advances, `execute_gripper_post_action()` reads contacts, validates/attaches the object, advances finger trajectories, and completes transitions. Task wrappers repeat this at 60 Hz until the expected terminal state or a timeout. The underlying controller is non-blocking, but the expert wrappers wait for grasp/release completion before the next task phase.

### Finger motion and validation

Joints 2 and 3 of configured fingers are controlled. Closed targets are `lower_limit + closure * joint_range`; open targets are zero clipped to joint limits. Interpolation starts from measured joint positions.

Only registered candidate bodies are eligible: moving cube for T1, bottle for new T2, lid for T4. Candidate selection counts distinct fingers touching eligible links and chooses the body with the greatest count. Validation combines required, mandatory, and supporting fingers with consecutive confirmation updates.

| Profile | Closure | Validation | Attachment |
|---|---|---|---|
| `t1_new_smooth` | Thumb/index/middle/ring sequentially, 0.4 s each | Thumb + at least two support fingers; 3 confirming updates after closure | Contacting fingers attached atomically |
| `t2_new_smooth` | All five simultaneously, 1.5 s | At least three constrained fingers; 3 confirming updates | Each finger attaches on contact |
| `t4_new_smooth` | All five simultaneously, 1.5 s; no contact freezing | At least two fingers via joint-2/3 links; 30 stable updates after closure | One rigid hand-to-lid anchor |

All three use nominal one-second opening. T1/T2 set thumb opposition to 1.209 radians. T1 holds the closed target on contact; T2 defaults to holding measured position. T4 completes closure, then confirms contact for 0.5 seconds.

T2 may validate before the full closing duration because its condition uses the constrained finger set. This differs from requiring three simultaneous unconstrained contacts. T1/T4 use current contacts after closure. Accumulated contacts, current contacts, and constrained fingers are distinct diagnostic quantities.

### Constraint-assisted manipulation

Fixed constraints preserve the current relative object/link pose at attachment. For a rigid hand anchor, the relative transform is `T_HO = inverse(T_WH) * T_WO`. The attachment itself does not teleport the object. Opening removes gripper-owned constraints before extension.

T4's single anchor avoids conflicting constraints between one object and multiple moving finger links. Transport remains constraint-assisted, rather than depending exclusively on frictional force closure. The post-action method does not continuously revalidate grasp quality in HOLDING.

Metadata records the profile used by T1, new T2, and T4. The optional `t1_sticky_fingertips` profile exists but is not selected by the reviewed T1 collection class.

Sources: [profiles](#L), [initialization](#L), [decoding](#L), [post-action logic](#L), [candidate selection](#L), [hand anchor](#L).

## 6. Task-specific expert algorithms

### T1: smooth stacking

T1 sets moving-cube mass to 0.2 kg and target-cube mass to 0.7 kg, selects the moving cube for head targeting, moves the unused shoulder away, and captures the left-arm hold posture.

1. Approach at a cube-relative offset `(0.13, 0.08, 0.15)` m with calibrated orientation.
2. Oppose the thumb, initialize the controller, and rotate locally by π/5.
3. Descend 0.045 m from the achieved pose, a calibration explicitly accounting for improved smooth-reference tracking.
4. Close sequentially; require HOLDING, otherwise open and abort.
5. Lift 0.11 m and align above the target with x/y offsets 0.055/0.06 m.
6. Level and descend using `level_object_to_table()`.
7. Disable robot collisions with both cubes and fix the lower cube to the world.
8. Weld the stack if the moving cube's AABB centre is supported within the target's x/y bounds and the absolute vertical AABB gap is ≤0.015 m.
9. Open the hand. If the initial weld failed, settle for 200 physics steps and retry; reject if still not weldable.
10. Retreat upward 0.25 m.

Welding preserves relative pose and zeros moving-cube velocity. It can occur before hand release, so the result is not an independently stable, unconstrained stack.

The leveling helper uses a separate Cartesian command followed by repeated 0.005 m descents, rather than minimum-jerk segments. Pre-grasp settling checks can warn and continue from achieved pose instead of rejecting.

The inherited final check requires `valid_grasp` and a moving-cube base height at least 0.03 m above the target base. Smooth execution adds weldability and release guards, but the final predicate alone does not prove physical support stability.

Sources: [smooth T1](../tasks/task_1_smooth.py), [inherited setup/success](../tasks/task_1.py), [placement helper](#L).

### T2: smooth bottle moving

Both smooth T2 variants use this sequence:

1. Set bottle/computer dynamics and smoothly prepare the left shoulder and thumb.
2. Construct a calibrated grasp pose offset `(0.03, 0.005, 0.06)` m from the bottle.
3. Construct a pre-grasp point 0.05 m back toward the current hand.
4. Move to pre-grasp, then approach with limits 0.02 m/s and 0.04 m/s², stopping on right-hand/bottle contact. Abort when not contact-terminated, no contact exists, and position error exceeds 0.02 m.
5. Grasp, lift 0.09 m, and execute a calibrated smooth arm-joint posture adjustment.
6. Set the bottle target to computer position plus `(0.25, 0, 0)` m: the code's world-coordinate interpretation of “left of”.
7. Transport at current bottle height, level it by removing roll/pitch while retaining yaw, then descend.
8. Stop descent on bottle contact with a body other than robot/computer, release, and retreat 0.11 m.

The hand target preserves the measured object-to-hand transform: `T_BH = inverse(T_WB) * T_WH`, followed by `T_WH_target = T_WB_target * T_BH`. Placement thus accounts for the actual grasp offset.

`t2_smooth` closes all five fingers with task-local quintic interpolation, freezes each at measured fingertip contact, and requires three accumulated contacted fingers. After 20 physics steps, `secure_grasp()` creates attachments. Release removes them, waits 120 steps, and opens smoothly.

`t2_new_smooth` replaces these routines with the shared binary controller while retaining the Cartesian phases and final predicate: valid grasp and final bottle x/y error ≤0.05 m. Uprightness, final height, long-term stability, and explicit table support are not checked.

Sources: [smooth T2](../tasks/task_2_smooth.py), [new gripper integration](../tasks/t2_new_smooth.py).

### T3: smooth pushing

T3 prepares an open hand and sets green-box mass to 0.5 kg and lateral friction to 0.5. The inherited target is the midpoint of hard-coded table bounds, `(-0.41, -0.0305)` m.

`get_push_alignment()` forms the push direction from the object's base x/y plus 0.05 m in both axes. It projects box-local x/y axes into the world horizontal plane, evaluates four outward face normals, and selects the smallest dot product with the push direction: the face opposing the desired movement. A face-specific rotation maps hand orientation to the box.

Four targets define elevated pre-push, contact start, push endpoint, and raised retreat. Horizontal stand-off is 0.08 m along the selected normal, with an additional 0.07 m x correction for the end-effector/palm offset. Push height is 0.02 m below the AABB top; pre-push/retreat are 0.1 m higher. All four segments use minimum-jerk timing and a minimum two-second duration.

There is no online corrective push loop or grasp attachment. Success requires final object-base x/y distance ≤0.06 m from the centre. The inherited nearby “5cm” comment differs from the executable 6 cm threshold.

Sources: [smooth T3](../tasks/task_3_smooth.py), [push geometry](#L), [target/success](../tasks/task_3.py).

### T4: smooth lid opening

Opening means removing and relocating a lid rather than rotating a hinge. T4 sets lid mass to 0.2 kg, prepares/holds the unused arm, approaches an object-relative pre-grasp, and then moves to a fixed world-space `GRASP_POSE`.

After contact-confirmed closure and single-hand attachment, it lifts 0.17 m. The placement target is box-base position offset -0.2 m in y. It aligns using hand x/y offsets 0.055/0.06 m, lowers 0.1 m, and releases. The sequence ends after opening with no final retreat.

Success requires valid grasp and final lid-base 3D distance strictly below 0.1 m from target. Settling failures warn but do not automatically abort. The fixed grasp pose limits adaptation to randomized object placements, despite the object-relative initial approach.

Source: [smooth T4](../tasks/task_4_smooth.py).

## 7. Recording schema and action meaning

The collector caches joint IDs/names, end-effector and eye links, and ten finger joints. At each sample it reads joint positions, EE pose, and tracked object base poses on the simulation thread, then queues a snapshot.

| File | Shape | Meaning |
|---|---|---|
| `timestamps.npy` | `(N,)` | Collector update count divided by 240 |
| `proprio.npy` | `(N,J)` | All exposed robot joint positions, float32 |
| `ee_poses.npy` | `(N,7)` | World x/y/z, roll/pitch/yaw, achieved hand closure |
| Raw `ee_deltas.npy` | `(N,7)` | Current minus preceding sampled EE vector; first row zero |
| `obj_poses.npy` | `(N,7K)` | Object base positions and x/y/z/w quaternions in metadata order |
| `joint_names.txt` | J lines | Proprioception column order |
| `ee_columns.txt` | Text | EE column labels |
| `camera_info.json` | JSON | Fixed-camera position, target, up vector |
| `metadata.json` | JSON | Task class, prompts, config path, objects, optional gripper profile |

J follows the loaded robot, not a collector constant. The reviewed tasks track two objects, producing 14 object-pose columns. Positions are in metres and rotations/joint angles in radians. The collector does not save joint velocities, torques, commanded reference poses, task phases, or contact diagnostics.

Measured closure is the clipped mean of normalized positions across available right-hand finger joints 2 and 3:

\[
c_t=\operatorname{clip}\left(\frac1M\sum_j\frac{q_{j,t}-q_j^{lo}}{q_j^{hi}-q_j^{lo}},0,1\right).
\]

Raw `ee_poses[:,6]` is achieved closure; raw `ee_deltas[:,6]` is its change. Neither is the binary command. Contacts can arrest fingers before full closure, so a fixed closure threshold is not a reliable general command decoder.

The queue holds 60 jobs. A full queue drops a sample rather than blocking physics. Since the previous EE pose is updated before queue insertion, a subsequent raw delta can refer to a dropped state. Uniform 5 Hz spacing and adjacent-state delta consistency must be verified in actual data.

Sources: [sampling](#L), [saving](#L), [state extraction](#L).

## 8. Offline image generation

`EpisodeRenderer` loads timestamps, proprioception, object poses, metadata and camera parameters, partitions frames into chunks, and uses multiprocessing with `spawn`. The CLI defaults to 15 workers; direct class construction defaults to six.

Each worker creates its own pyCub scene. An initialization lock prevents concurrent generated-asset writes, a dummy visualizer suppresses Open3D windows, and the script relaunches under Xvfb when no display exists.

For each frame, robot joint positions and tracked object base poses are reset to saved values. Physics is not stepped. TinyRenderer produces right-eye, left-eye and fixed-camera RGB images, normally 320×240 pixels with shadows. This is state reconstruction rather than dynamic replay.

Workers save temporary compressed chunks. The parent sorts them by frame index and writes `<camera>_video.npz` with a `video` array. The decoder creates PNG sequences such as `frame_00000.png`. `process_dataset.py` processes immediate episode subdirectories, so its normal input is a session directory rather than a root containing sessions.

Reconstruction depends on current assets/configuration and restores only recorded robot joints and tracked object base poses, not every possible articulated object or simulator state. Render errors omit images without inserting placeholders or enforcing frame-count equality. Camera lengths require validation before packaging.

Optional online collector rendering reads the live world later in its worker and does not restore the queued state. Exact online image/state synchronization is therefore not guaranteed. Normal state-only collection followed by offline reconstruction avoids that particular issue.

Sources: [renderer](../tools/render_recording.py), [batch orchestration](../process_dataset.py), [decoder](../tools/decoder.py).

## 9. Training-label conversion and export

### Next-action alignment

Raw delta row t describes motion from t-1 to t. Next-action learning instead needs observation t paired with motion to t+1.

`make_next_action_dataset.py` computes N-1 labels from N poses. Translation is `p[t+1]-p[t]`. Orientation uses a world-frame relative rotation:

\[
R_{delta,t}=R_{t+1}R_t^{-1},\qquad R_{t+1}=R_{delta,t}R_t,
\]

then represents it with xyz Euler angles. This differs from the raw collector's componentwise wrapped Euler subtraction. The script preserves the old actions as `ee_deltas_wrong_time.npy` and writes conversion metadata.

Observation arrays and videos retain N frames. Packaging must use N-1 actions and pair action i with observation i, excluding the final observation from action-bearing records. Labels are measured displacements over nominal 0.2-second intervals, not velocities or original 60 Hz targets.

### Absolute gripper labels

`make_absolute_gripper_actions.py` median-smooths measured closure, detects sustained signed motion, backdates each confirmed transition to its first significant movement, and latches a command until the next opposite event.

Defaults are five-sample smoothing, five-sample motion windows, two confirmation windows, and ten actions between transitions. Threshold is the greater of 0.003 and 5% of the 5th-to-95th percentile closure range. Initial command can be inferred or specified; manual transition indices are supported.

It produces N-1 commands. With `--write`, it saves the command array and diagnostic JSON, preserves `ee_deltas_not_absolute_gripper.npy`, and replaces action column 6 when the action array has the expected N-1 rows. Optional plots expose detected transitions.

The final seven-dimensional action contains world translation delta, world-relative rotation represented as Euler angles, and an absolute binary gripper command.

The preparation order is **next-action conversion, then binary gripper conversion**. Recomputing next-actions afterwards restores closure differences in column 6. The older `fix_orientation.py` retains N rows and backward alignment, so it is not a substitute for next-action conversion.

Binary labels are inferred from achieved motion, not logged expert commands. Their timing should be inspected, particularly for nearly constant signals and open-hand T3 trajectories.

### Export boundary

`convert_lerobot.py` consumes an already prepared TFDS/RLDS train split. It maps three camera streams, proprioception, EE pose, object pose, timestamps, actions, reward, discount and language into LeRobot features, infers dimensions, saves episodes, finalizes, and optionally reloads a sample for validation. Upload is optional and disabled by default.

This is an RLDS-to-LeRobot stage. The reviewed collection pipeline does not invoke a raw-to-RLDS builder; the inspected conversion scripts do not establish that builder's reward or terminal-label generation. Those details must come from the actual external builder used for a dataset release.

The converter defaults to 10 fps, whereas collection is 5 Hz. Unresampled recordings should use 5 fps export metadata. Preserving timestamps does not automatically correct an inconsistent declared rate.

Sources: [next-action conversion](../scripts/make_next_action_dataset.py), [binary gripper conversion](../scripts/make_absolute_gripper_actions.py), [older rotation correction](../scripts/fix_orientation.py), [LeRobot conversion](../scripts/convert_lerobot.py).

## 10. Diversity, language, and reproducibility

Randomization is inherited and runs only when requested. T1 samples cubes in separate regions; T2 varies bottle/laptop positions; T3 changes box and distractor placement; T4 moves base/lid together in a narrow region. These are task-specific routines, not a general domain-randomization framework.

Camera randomization needs both runner randomization and the YAML toggle. All checked T1–T4 configurations disable the camera toggle. Enabled routines overwrite persistent camera configuration in place, so successive offsets can accumulate instead of staying around an immutable baseline.

`--randomizer` uses `type=bool`; a nonempty string such as `False` evaluates as true. Omitting it retains the false default.

`generate_llm_variations()` is a placeholder returning the correct base prompt plus four fixed stacking paraphrases. There is no LLM call. Since `BaseTask` randomly selects one, non-stacking tasks can receive incorrect stacking instructions. `augment_prompts.py` replaces prompts from an external text list, but its existence does not establish that stored episodes were corrected.

`augment_cameras.py` overwrites saved camera metadata with a baseline and x/y jitter. Images must be rerendered after this change. Neither augmentation is automatically invoked by collection.

Metadata does not preserve random seeds, source revision, dependency versions, a complete immutable configuration, or all transformation history. Conversion JSON files and backups improve traceability but do not replace a full dataset provenance manifest.

Sources: [prompt placeholder](#L), [metadata](#L), [prompt augmentation](../scripts/augment_prompts.py), [camera augmentation](../scripts/augment_cameras.py), inherited randomizers in the parent task modules.

## 11. Thesis interpretation and verification boundaries

A source-supported description is:

> Demonstrations are generated by scripted, privileged-state manipulation experts operating an iCub model in PyBullet. The experts sequence task-specific Cartesian and joint targets using quintic minimum-jerk timing and quaternion interpolation, generally issuing references at 60 Hz over a 240 Hz simulation. Contact-aware hand controllers validate calibrated grasps and stabilize transport with fixed constraints. Successful attempts are sampled at 5 Hz, storing achieved robot and object states for offline multi-camera rendering. Subsequent processing aligns actions with the next observation, computes world-frame relative rotations, and converts measured hand closure into latched binary commands.

This description needs four qualifications: smoothness applies to reference segments and is incomplete in T1 placement; grasping/stacking use simulator assistance; success means passing task-specific geometric predicates rather than a common stability test; and rendering, label conversion, language correction and export are separate operations whose execution must be checked for each dataset.

For an empirical dataset assessment, the missing measurements are episode counts and attempt success rates, timestamp/frame consistency, achieved tracking and jerk statistics, correctness of language and gripper transitions, and unconstrained/replay performance. None of these measurements was established by this static review. Implementation files and existing recordings were left unchanged; only this report was created.

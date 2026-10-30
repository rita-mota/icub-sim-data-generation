"""Smooth T5 wiping expert; preserves the original palm/sponge mechanics."""

import json
import math
import os

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose
from scipy.spatial.transform import Rotation as R, Slerp

from tasks.task_5 import T5Cleaning


class T5SmoothCleaning(T5Cleaning):
    """Minimum-jerk approach, wiping strokes, mop-up, and retreat."""

    CONTROL_HZ = 60.0
    PHYSICS_SUBSTEPS = 4  # 240 Hz physics, 5 Hz dataset sampling.
    MAX_LINEAR_SPEED = 0.12
    MAX_LINEAR_ACCELERATION = 0.25
    MAX_ANGULAR_SPEED = 1.2
    MAX_ANGULAR_ACCELERATION = 2.0
    LEFT_ARM_JOINTS = (
        "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
        "l_wrist_prosup", "l_wrist_pitch", "l_wrist_yaw",
    )

    def setup_environment(self):
        super().setup_environment()
        self.valid_clean = False
        self._cleaning = False
        self._left_arm_targets = {}
        # The sponge is pressed with an open hand, not a binary finger grasp.
        # Establish this posture before recording starts.
        self.skills.open_hand()

    @staticmethod
    def _minimum_jerk(u):
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def _step_physics(self, steps=PHYSICS_SUBSTEPS):
        for _ in range(steps):
            self.client.update_simulation(None)
            self.skills.track_target_with_head()
            if self._cleaning:
                self.clean_step()

    def _joint_position(self, name):
        handle = self.client.get_joint_handle(name)
        return p.getJointState(
            self.client.robot, handle.robot_joint_id,
            physicsClientId=self.client._client,
        )[0]

    def _capture_left_arm(self):
        self._left_arm_targets = {
            name: self._joint_position(name) for name in self.LEFT_ARM_JOINTS
        }

    def _move_joints(self, targets, duration=1.5):
        starts = {name: self._joint_position(name) for name in targets}
        steps = max(1, math.ceil(duration * self.CONTROL_HZ))
        for step in range(1, steps + 1):
            blend = self._minimum_jerk(step / steps)
            for name, target in targets.items():
                self.client.move_position(
                    name, starts[name] + blend * (target - starts[name]),
                    wait=False, check_collision=False, velocity=3.0,
                    timeout=None,
                )
            self._step_physics()
        self._capture_left_arm()

    def _command_pose(self, pose):
        self.client.move_cartesian(
            pose, wait=False, timeout=None, check_collision=False,
        )
        # Full-body IK must not overwrite the deliberately positioned left arm.
        if self._left_arm_targets:
            self.client.move_position(
                list(self._left_arm_targets),
                list(self._left_arm_targets.values()),
                wait=False, check_collision=False, velocity=3.0, timeout=None,
            )
        self._step_physics()

    def _move_pose(self, target, minimum_duration=2.0):
        start = self.client.end_effector.get_pose()
        start_pos = np.asarray(start.pos, dtype=float)
        target_pos = np.asarray(target.pos, dtype=float)
        rotations = R.from_quat([start.ori, target.ori])
        angle = (rotations[1] * rotations[0].inv()).magnitude()
        distance = np.linalg.norm(target_pos - start_pos)
        # Analytical peak derivatives of quintic minimum-jerk time scaling.
        acceleration_scale = 10.0 / math.sqrt(3.0)
        duration = max(
            minimum_duration,
            1.875 * distance / self.MAX_LINEAR_SPEED,
            math.sqrt(acceleration_scale * distance / self.MAX_LINEAR_ACCELERATION),
            1.875 * angle / self.MAX_ANGULAR_SPEED,
            math.sqrt(acceleration_scale * angle / self.MAX_ANGULAR_ACCELERATION),
        )
        steps = max(1, math.ceil(duration * self.CONTROL_HZ))
        orientation_path = Slerp([0.0, 1.0], rotations)
        for step in range(1, steps + 1):
            blend = self._minimum_jerk(step / steps)
            self._command_pose(Pose(
                (start_pos + blend * (target_pos - start_pos)).tolist(),
                orientation_path([blend]).as_quat()[0].tolist(),
            ))

    def _settle_pose(self, target, duration=0.5, tolerance=0.008):
        for _ in range(math.ceil(duration * self.CONTROL_HZ)):
            actual = self.client.end_effector.get_pose()
            if np.linalg.norm(np.asarray(actual.pos) - target.pos) <= tolerance:
                return True
            self._command_pose(target)
        return False

    def _prepare_strip(self):
        # Preserve the original per-strip posture changes, but interpolate them.
        self._move_joints({
            "l_elbow": 1.8,
            "r_hand_thumb_0_joint": 0.0,
            "l_shoulder_roll": self._joint_position("l_shoulder_roll") + 0.3,
        }, duration=0.75)

    def execute_expert(self):
        self.skills.set_object_to_track(self.sponge_id)
        self._move_joints({
            "l_shoulder_pitch": -1.6,
            "l_shoulder_roll": 2.1,
            "l_elbow": 1.8,
        })
        sponge_pos = p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client,
        )[0]
        flat_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        print("[T5 smooth] Approaching sponge.")
        approach = Pose(
            [sponge_pos[0] + 0.05, sponge_pos[1], sponge_pos[2] + 0.1],
            flat_orn,
        )
        self._move_pose(approach)
        self._settle_pose(approach)

        target_z = sponge_pos[2] + 0.035
        current = self.client.end_effector.get_pose()
        descent = Pose([current.pos[0], current.pos[1], target_z], current.ori)
        self._move_pose(descent, minimum_duration=1.5)
        self._settle_pose(descent)
        self._sponge_table_z = p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client,
        )[0][2]
        self._step_physics(30)

        # Original fallback press: three 3 mm downward movements, now smooth.
        if not self.skills.active_constraints:
            current = self.client.end_effector.get_pose()
            press = Pose(
                [current.pos[0], current.pos[1], current.pos[2] - 0.009],
                current.ori,
            )
            self._move_pose(press, minimum_duration=1.0)
            self._step_physics(60)
            self.skills.secure_grasp_all_contacts(self.sponge_id)

        self._sponge_table_z = p.getBasePositionAndOrientation(
            self.sponge_id, physicsClientId=self.client._client,
        )[0][2]
        self.skills.attach_sponge_to_palm(self.sponge_id, self._sponge_table_z)
        self.valid_clean = True
        self._cleaning = True
        current_orn = self.client.end_effector.get_pose().ori

        if self.spill_particles:
            positions = np.asarray([d["pos"] for d in self.spill_particles.values()])
            low, high = positions.min(axis=0), positions.max(axis=0)
            x_start, x_end = low[0] - 0.03, high[0] + 0.03
            strip_ys = np.linspace(low[1] - 0.02, high[1] + 0.02, 6)
            for index, y in enumerate(strip_ys):
                self._prepare_strip()
                if not self.spill_particles:
                    break
                x_from, x_to = (x_start, x_end) if index % 2 == 0 else (x_end, x_start)
                self._move_pose(Pose([x_from, y, target_z], current_orn))
                self._step_physics(15)
                if self.spill_particles:
                    # Finish the stroke even if its last particle disappears
                    # mid-path; truncating it would introduce an abrupt stop.
                    self._move_pose(Pose([x_to, y, target_z], current_orn))
                print(f"[T5 smooth] Strip {index + 1}/6: {len(self.spill_particles)} remaining.")
                if not self.spill_particles:
                    break

        remaining = sorted(self.spill_particles.items(), key=lambda item: item[1]["pos"][0])
        for name, data in remaining:
            if not self.spill_particles:
                break
            if name not in self.spill_particles:
                continue
            self._prepare_strip()
            if not self.spill_particles:
                break
            x, y, _ = data["pos"]
            self._move_pose(Pose([x, y, target_z], current_orn))
            self._step_physics(20)

        print("[T5 smooth] Releasing sponge and retreating.")
        self._cleaning = False
        self.skills.release_grasp()
        current = self.client.end_effector.get_pose()
        self._move_pose(Pose(
            [current.pos[0], current.pos[1], current.pos[2] + 0.12], current.ori,
        ))
        if getattr(self.collector, "save_dir", None):
            os.makedirs(self.collector.save_dir, exist_ok=True)
            with open(os.path.join(self.collector.save_dir, "spill_data.json"), "w") as file:
                json.dump(self.spill_log, file)


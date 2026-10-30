"""Smooth Task 4 expert using the shared binary gripper controller."""

import math

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from core.robot_skills import GripperState
from tasks.task_4 import T4Opening


class T4SmoothOpening(T4Opening):
    """Open the box with smooth Cartesian references and replayable grip commands."""

    GRIPPER_PROFILE = "t4_new_smooth"

    SIM_HZ = 240.0
    CONTROL_HZ = 60.0
    PHYSICS_SUBSTEPS = int(SIM_HZ / CONTROL_HZ)

    MAX_LINEAR_SPEED = 0.08
    MAX_LINEAR_ACCELERATION = 0.15
    MAX_ANGULAR_SPEED = 0.8
    MAX_ANGULAR_ACCELERATION = 1.2

    LEFT_ARM_JOINTS = (
        "l_shoulder_pitch",
        "l_shoulder_roll",
        "l_shoulder_yaw",
        "l_elbow",
        "l_wrist_prosup",
        "l_wrist_pitch",
        "l_wrist_yaw",
    )

    GRASP_POSE = Pose(
        [-0.31255515336990356, 0.1912525188922882, 0.1073146212100983],
        [
            0.7538662552833557,
            -0.24759431183338165,
            0.2670873999595642,
            0.5468518137931824,
        ],
    )

    @staticmethod
    def _minimum_jerk(progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        return 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5

    def _step_simulation(self, steps=None):
        for _ in range(steps or self.PHYSICS_SUBSTEPS):
            self.client.update_simulation(None)

    def _joint_position(self, joint_name):
        handle = self.client.get_joint_handle(joint_name)
        return p.getJointState(
            self.client.robot,
            handle.robot_joint_id,
            physicsClientId=self.client._client,
        )[0]

    def _move_joints_minimum_jerk(self, targets, duration=1.5):
        starts = {name: self._joint_position(name) for name in targets}
        steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        for step in range(1, steps + 1):
            blend = self._minimum_jerk(step / steps)
            for name, target in targets.items():
                self.client.move_position(
                    name,
                    starts[name] + blend * (target - starts[name]),
                    wait=False,
                    check_collision=False,
                    velocity=3.0,
                    timeout=None,
                )
            self._step_simulation()

    def _capture_left_arm_hold(self):
        self._left_arm_hold_targets = {
            name: self._joint_position(name)
            for name in self.LEFT_ARM_JOINTS
        }

    def _hold_left_arm(self):
        self.client.move_position(
            list(self._left_arm_hold_targets),
            list(self._left_arm_hold_targets.values()),
            wait=False,
            check_collision=False,
            velocity=3.0,
            timeout=None,
        )

    def _command_cartesian(self, pose):
        self.client.move_cartesian(
            pose,
            wait=False,
            timeout=None,
            check_collision=False,
        )
        self._hold_left_arm()
        self._step_simulation()

    def _trajectory_duration(self, start_pose, target_pose, minimum_duration):
        distance = float(np.linalg.norm(
            np.asarray(target_pose.pos) - np.asarray(start_pose.pos)
        ))
        start_rotation = R.from_quat(start_pose.ori)
        target_rotation = R.from_quat(target_pose.ori)
        angle = float((target_rotation * start_rotation.inv()).magnitude())
        speed_scale = 1.875
        acceleration_scale = 10.0 / math.sqrt(3.0)
        return max(
            minimum_duration,
            speed_scale * distance / self.MAX_LINEAR_SPEED,
            math.sqrt(
                acceleration_scale * distance / self.MAX_LINEAR_ACCELERATION
            ),
            speed_scale * angle / self.MAX_ANGULAR_SPEED,
            math.sqrt(
                acceleration_scale * angle / self.MAX_ANGULAR_ACCELERATION
            ),
        )

    def _move_cartesian_minimum_jerk(self, target_pose, minimum_duration=1.2):
        start_pose = self.client.end_effector.get_pose()
        start_position = np.asarray(start_pose.pos, dtype=float)
        target_position = np.asarray(target_pose.pos, dtype=float)
        orientation_path = Slerp(
            [0.0, 1.0],
            R.from_quat([start_pose.ori, target_pose.ori]),
        )
        duration = self._trajectory_duration(
            start_pose,
            target_pose,
            minimum_duration,
        )
        steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        for step in range(1, steps + 1):
            blend = self._minimum_jerk(step / steps)
            position = start_position + blend * (
                target_position - start_position
            )
            orientation = orientation_path([blend]).as_quat()[0]
            self._command_cartesian(
                Pose(position.tolist(), orientation.tolist())
            )

    def _settle_at_pose(self, target_pose, duration=0.75, tolerance=0.008):
        target = np.asarray(target_pose.pos, dtype=float)
        steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        for _ in range(steps):
            actual = np.asarray(
                self.client.end_effector.get_pose().pos,
                dtype=float,
            )
            if np.linalg.norm(target - actual) <= tolerance:
                return True
            self._command_cartesian(target_pose)
        return False

    def _move_and_settle(self, pose, minimum_duration, settle_duration=0.75):
        self._move_cartesian_minimum_jerk(pose, minimum_duration)
        return self._settle_at_pose(pose, settle_duration)

    @staticmethod
    def _offset_pose(pose, dx=0.0, dy=0.0, dz=0.0):
        return Pose(
            [pose.pos[0] + dx, pose.pos[1] + dy, pose.pos[2] + dz],
            pose.ori,
        )

    def _run_gripper_command(self, command, timeout):
        self.skills.execute_gripper_pre_action(command)
        terminal_states = (
            {GripperState.HOLDING, GripperState.CLOSED_EMPTY}
            if command == 0
            else {GripperState.OPEN}
        )
        diagnostics = None
        steps = max(
            1,
            int(math.ceil(timeout * self.skills.gripper_control_hz)),
        )
        for _ in range(steps):
            self._step_simulation()
            diagnostics = self.skills.execute_gripper_post_action(command)
            if self.skills.gripper_state in terminal_states:
                break
        return self.skills.gripper_state, diagnostics

    def _initialize_gripper(self):
        state = self.skills.initialize_gripper_controller(
            candidate_object_ids=[self.box_top_id],
            hand="right",
            profile=self.GRIPPER_PROFILE,
            initial_command=1,
        )
        if state == GripperState.OPEN:
            return True
        self.client.logger.error(
            f"T4 gripper did not initialize open: {state.name}."
        )
        return False

    def _grasp_lid(self):
        state, diagnostics = self._run_gripper_command(
            0,
            self.skills.gripper_closing_duration + 1.0,
        )
        if state == GripperState.HOLDING:
            self.client.logger.info(
                "T4 lid secured after settled closure: "
                f"contacting_fingers="
                f"{diagnostics['current_contacting_fingers']}, "
                f"constraint_link_ids="
                f"{diagnostics['constraint_link_ids']}."
            )
            return True
        self.client.logger.error(
            "T4 gripper failed to grasp the lid: "
            f"state={state.name}, diagnostics={diagnostics}."
        )
        self._run_gripper_command(1, 2.0)
        return False

    def _release_lid(self):
        state, diagnostics = self._run_gripper_command(
            1,
            self.skills.gripper_opening_duration + 1.0,
        )
        if state == GripperState.OPEN:
            return True
        self.client.logger.error(
            "T4 gripper failed to open: "
            f"state={state.name}, diagnostics={diagnostics}."
        )
        return False

    def execute_expert(self):
        self.valid_grasp = False
        self.skills.set_dynamics(self.box_top_id, mass=0.2)
        self.skills.set_object_to_track(self.box_top_id)

        self._move_joints_minimum_jerk(
            {"l_shoulder_pitch": -1.6},
            duration=1.5,
        )
        self._capture_left_arm_hold()
        self.skills.track_target_with_head()

        box_top_pos = self.client.getBasePositionAndOrientation(
            self.box_top_id
        )[0]
        pregrasp_pose = Pose(
            [box_top_pos[0] + 0.13, box_top_pos[1] + 0.08, box_top_pos[2] + 0.2],
            p.getQuaternionFromEuler([1.5, 0.0, 0.0]),
        )
        if not self._move_and_settle(pregrasp_pose, 2.0, 1.0):
            self.client.logger.warning("T4 pre-grasp pose did not fully settle.")

        if not self._move_and_settle(self.GRASP_POSE, 2.0, 1.0):
            self.client.logger.warning("T4 grasp pose did not fully settle.")
        if not self._initialize_gripper() or not self._grasp_lid():
            return
        self.valid_grasp = True

        current = self.client.end_effector.get_pose()
        self._move_cartesian_minimum_jerk(
            self._offset_pose(current, dz=0.17),
            minimum_duration=2.0,
        )

        base_pos = self.client.getBasePositionAndOrientation(
            self.box_bottom_id
        )[0]
        self.target_final_pos = np.array(
            [base_pos[0], base_pos[1] - 0.2, base_pos[2]],
            dtype=float,
        )
        current = self.client.end_effector.get_pose()
        alignment_pose = Pose(
            [
                self.target_final_pos[0] + 0.055,
                self.target_final_pos[1] + 0.06,
                current.pos[2],
            ],
            current.ori,
        )
        self._move_cartesian_minimum_jerk(
            alignment_pose,
            minimum_duration=2.5,
        )

        current = self.client.end_effector.get_pose()
        self._move_cartesian_minimum_jerk(
            self._offset_pose(current, dz=-0.1),
            minimum_duration=1.5,
        )

        if not self._release_lid():
            self.valid_grasp = False
            return

    def check_success(self):
        if not self.valid_grasp:
            return False
        top_pos = np.asarray(
            self.client.getBasePositionAndOrientation(self.box_top_id)[0],
            dtype=float,
        )
        distance = float(np.linalg.norm(top_pos - self.target_final_pos))
        if distance < 0.1:
            print(f"Success! Box is within {distance:.3f} meters of the target.")
            return True
        print(f"Failure. Box is {distance:.3f} meters away from the target.")
        return False

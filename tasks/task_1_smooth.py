"""Smooth Task 1 expert using the shared binary gripper state machine.

This task expects RobotSkills to provide a ``t1_new_smooth`` profile with a
sequential thumb/index/middle/ring close and an atomic, T1-specific grasp
predicate (thumb plus at least two supporting fingers).
"""

import math

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from core.robot_skills import GripperState
from tasks.task_1 import T1Stacking


class T1SmoothStacking(T1Stacking):
    """Stack the cubes with smooth references and the shared hand controller."""

    GRIPPER_PROFILE = "t1_new_smooth"

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

    # The original blocking Cartesian motion undershoots its requested
    # 0.055 m descent.  The smooth controller follows that target more
    # closely, which places the hand too low for the opposed thumb.  Use the
    # original task's achieved motion (about 0.045 m) as the smooth target.
    GRASP_DESCENT = 0.045

    @staticmethod
    def _minimum_jerk(progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        return (
            10.0 * progress**3
            - 15.0 * progress**4
            + 6.0 * progress**5
        )

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

    def _body_pose(self, body_id):
        return p.getBasePositionAndOrientation(
            body_id,
            physicsClientId=self.client._client,
        )

    def _body_aabb(self, body_id):
        return p.getAABB(body_id, physicsClientId=self.client._client)

    def _move_joints_minimum_jerk(self, targets, duration=1.5):
        starts = {
            name: self._joint_position(name)
            for name in targets
        }
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))

        for step in range(1, control_steps + 1):
            blend = self._minimum_jerk(step / control_steps)
            for name, target in targets.items():
                position = starts[name] + blend * (target - starts[name])
                self.client.move_position(
                    name,
                    position,
                    wait=False,
                    check_collision=False,
                    velocity=3.0,
                    timeout=None,
                )
            self._step_simulation()

    def _capture_left_arm_hold(self):
        """Preserve the unused arm against whole-body IK side effects."""
        self._left_arm_hold_targets = {
            name: self._joint_position(name)
            for name in self.LEFT_ARM_JOINTS
        }

    def _hold_left_arm(self):
        targets = getattr(self, "_left_arm_hold_targets", None)
        if not targets:
            return
        self.client.move_position(
            list(targets),
            list(targets.values()),
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

    @staticmethod
    def _offset_z(pose, dz):
        return Pose(
            [pose.pos[0], pose.pos[1], pose.pos[2] + dz],
            pose.ori,
        )

    def _trajectory_duration(self, start_pose, target_pose, minimum_duration):
        start_pos = np.asarray(start_pose.pos, dtype=float)
        target_pos = np.asarray(target_pose.pos, dtype=float)
        linear_distance = float(np.linalg.norm(target_pos - start_pos))
        start_rotation = R.from_quat(start_pose.ori)
        target_rotation = R.from_quat(target_pose.ori)
        angular_distance = float(
            (target_rotation * start_rotation.inv()).magnitude()
        )
        speed_scale = 1.875
        acceleration_scale = 10.0 / math.sqrt(3.0)
        linear_time = max(
            speed_scale * linear_distance / self.MAX_LINEAR_SPEED,
            math.sqrt(
                acceleration_scale
                * linear_distance
                / self.MAX_LINEAR_ACCELERATION
            ),
        )
        angular_time = max(
            speed_scale * angular_distance / self.MAX_ANGULAR_SPEED,
            math.sqrt(
                acceleration_scale
                * angular_distance
                / self.MAX_ANGULAR_ACCELERATION
            ),
        )
        return max(
            minimum_duration,
            linear_time,
            angular_time,
        )

    def _move_cartesian_minimum_jerk(
        self,
        target_pose,
        minimum_duration=1.2,
    ):
        start_pose = self.client.end_effector.get_pose()
        start_pos = np.asarray(start_pose.pos, dtype=float)
        target_pos = np.asarray(target_pose.pos, dtype=float)
        rotations = R.from_quat(
            np.stack(
                [
                    np.asarray(start_pose.ori, dtype=float),
                    np.asarray(target_pose.ori, dtype=float),
                ]
            )
        )
        orientation_path = Slerp([0.0, 1.0], rotations)
        duration = self._trajectory_duration(
            start_pose,
            target_pose,
            minimum_duration,
        )
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))

        for step in range(1, control_steps + 1):
            blend = self._minimum_jerk(step / control_steps)
            position = start_pos + blend * (target_pos - start_pos)
            orientation = orientation_path([blend]).as_quat()[0]
            self._command_cartesian(
                Pose(position.tolist(), orientation.tolist())
            )

    def _settle_at_pose(
        self,
        target_pose,
        duration=0.75,
        position_tolerance=0.006,
        orientation_tolerance=0.04,
    ):
        """Hold one Cartesian reference until tracking error is acceptable."""
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        target_position = np.asarray(target_pose.pos, dtype=float)
        target_orientation = np.asarray(target_pose.ori, dtype=float)

        for _ in range(control_steps):
            actual_pose = self.client.end_effector.get_pose()
            actual_position = np.asarray(actual_pose.pos, dtype=float)
            actual_orientation = np.asarray(actual_pose.ori, dtype=float)
            position_error = float(
                np.linalg.norm(target_position - actual_position)
            )
            quaternion_dot = float(np.clip(
                abs(np.dot(target_orientation, actual_orientation)),
                0.0,
                1.0,
            ))
            orientation_error = 2.0 * math.acos(quaternion_dot)
            if (
                position_error <= position_tolerance
                and orientation_error <= orientation_tolerance
            ):
                return True

            self._command_cartesian(target_pose)

        return False

    def _move_and_settle(
        self,
        target_pose,
        minimum_duration,
        settle_duration=0.75,
        position_tolerance=0.006,
        orientation_tolerance=0.04,
    ):
        self._move_cartesian_minimum_jerk(
            target_pose,
            minimum_duration=minimum_duration,
        )
        return self._settle_at_pose(
            target_pose,
            duration=settle_duration,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )

    def _log_grasp_alignment(self, label):
        cube_position = np.asarray(self._body_pose(self.cube_id)[0])
        hand_position = np.asarray(
            self.client.end_effector.get_pose().pos,
        )
        offset = hand_position - cube_position
        self.client.logger.info(
            f"T1 {label} hand-minus-cube offset: "
            f"x={offset[0]:.4f}, y={offset[1]:.4f}, z={offset[2]:.4f}."
        )

    def _run_gripper_command(self, command, timeout):
        self.skills.execute_gripper_pre_action(command)
        control_steps = max(
            1,
            int(math.ceil(timeout * self.skills.gripper_control_hz)),
        )
        terminal_states = (
            {GripperState.HOLDING, GripperState.CLOSED_EMPTY}
            if command == 0
            else {GripperState.OPEN}
        )
        diagnostics = None

        for _ in range(control_steps):
            self._step_simulation()
            diagnostics = self.skills.execute_gripper_post_action(command)
            if self.skills.gripper_state in terminal_states:
                break

        return self.skills.gripper_state, diagnostics

    def _initialize_gripper(self):
        state = self.skills.initialize_gripper_controller(
            candidate_object_ids=[self.cube_id],
            hand="right",
            profile=self.GRIPPER_PROFILE,
            initial_command=1,
        )
        if state != GripperState.OPEN:
            self.client.logger.error(
                "T1 shared gripper did not initialize open: "
                f"{state.name}."
            )
            return False
        return True

    def _gripper_reaches(self, command, timeout, expected, action):
        state, diagnostics = self._run_gripper_command(command, timeout)
        if state != expected:
            self.client.logger.error(
                f"T1 shared gripper failed to {action}: "
                f"state={state.name}, diagnostics={diagnostics}."
            )
            return False
        return True

    def _grasp_cube(self):
        if self._gripper_reaches(
            0,
            4.0,
            GripperState.HOLDING,
            "grasp the cube",
        ):
            return True
        self._run_gripper_command(1, 2.0)
        return False

    def _release_cube(self):
        return self._gripper_reaches(1, 2.0, GripperState.OPEN, "open")

    def _nail_target_cube(self):
        target_pos, target_orn = self._body_pose(self.target_cube_id)
        self.client.nail_cid = p.createConstraint(
            parentBodyUniqueId=self.target_cube_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=target_pos,
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=target_orn,
            physicsClientId=self.client._client,
        )
        print("[T1 smooth] Bottom cube is now nailed to the table.")

    def _weld_stack_if_aligned(self, phase):
        moving_min, moving_max = self._body_aabb(self.cube_id)
        target_min, target_max = self._body_aabb(self.target_cube_id)
        moving_center_x = 0.5 * (moving_min[0] + moving_max[0])
        moving_center_y = 0.5 * (moving_min[1] + moving_max[1])
        center_supported = (
            target_min[0] <= moving_center_x <= target_max[0]
            and target_min[1] <= moving_center_y <= target_max[1]
        )
        vertical_gap = moving_min[2] - target_max[2]
        vertically_aligned = abs(vertical_gap) <= 0.015
        if not (center_supported and vertically_aligned):
            self.client.logger.warning(
                f"T1 stack not weldable during {phase}: "
                f"center_supported={center_supported}, "
                f"vertical_gap={vertical_gap:.4f} m."
            )
            return False

        target_pos, target_orn = self._body_pose(self.target_cube_id)
        cube_pos, cube_orn = self._body_pose(self.cube_id)
        inverse_target_pos, inverse_target_orn = p.invertTransform(
            target_pos,
            target_orn,
        )
        relative_pos, relative_orn = p.multiplyTransforms(
            inverse_target_pos,
            inverse_target_orn,
            cube_pos,
            cube_orn,
        )
        self.client.weld_cid = p.createConstraint(
            parentBodyUniqueId=self.target_cube_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.cube_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=relative_orn,
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client._client,
        )
        p.resetBaseVelocity(
            self.cube_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.client._client,
        )
        print(
            f"[T1 smooth] Top cube welded during {phase} "
            f"without changing its pose."
        )
        return True

    def _settle_and_weld_stack(self):
        """Fallback for a cube that needed a short natural settling period."""
        self._step_simulation(200)
        return self._weld_stack_if_aligned("post-release settling")

    def execute_expert(self):
        self.valid_grasp = False
        self.skills.set_dynamics(self.cube_id, mass=0.2)
        self.skills.set_dynamics(self.target_cube_id, mass=0.7)
        self.skills.set_object_to_track(self.cube_id)

        self._move_joints_minimum_jerk(
            {"l_shoulder_pitch": -1.6},
            duration=1.5,
        )
        self._capture_left_arm_hold()
        self.skills.track_target_with_head()

        cube_pos = self._body_pose(self.cube_id)[0]
        pregrasp_pose = Pose(
            [cube_pos[0] + 0.13, cube_pos[1] + 0.08, cube_pos[2] + 0.15],
            p.getQuaternionFromEuler([1.5, 0.0, 0.0]),
        )
        reached_pregrasp = self._move_and_settle(
            pregrasp_pose,
            minimum_duration=2.0,
            settle_duration=1.0,
        )
        if not reached_pregrasp:
            self.client.logger.warning(
                "T1 did not fully settle at the original pre-grasp pose; "
                "continuing from the achieved pose as the original task does."
            )
        self._log_grasp_alignment("pre-grasp")

        self._move_joints_minimum_jerk(
            {"r_hand_thumb_0_joint": 1.209},
            duration=0.75,
        )
        if not self._initialize_gripper():
            return

        ee_state = p.getLinkState(
            self.client.robot,
            self.client.end_effector.link_id,
            computeForwardKinematics=True,
            physicsClientId=self.client._client,
        )
        rotation_position = ee_state[0]
        rotation_orientation = p.multiplyTransforms(
            [0, 0, 0],
            ee_state[1],
            [0, 0, 0],
            p.getQuaternionFromEuler([0.0, 0.0, np.pi / 5]),
        )[1]
        rotation_pose = Pose(
            list(rotation_position),
            list(rotation_orientation),
        )
        reached_rotation = self._move_and_settle(
            rotation_pose,
            minimum_duration=1.0,
            settle_duration=0.75,
        )
        if not reached_rotation:
            self.client.logger.warning(
                "T1 did not fully settle after the local wrist rotation; "
                "descending from the achieved pose."
            )

        achieved_rotation_pose = self.client.end_effector.get_pose()
        grasp_pose = self._offset_z(
            achieved_rotation_pose,
            -self.GRASP_DESCENT,
        )
        reached_grasp_pose = self._move_and_settle(
            grasp_pose,
            minimum_duration=1.0,
            settle_duration=1.0,
        )
        if not reached_grasp_pose:
            self.client.logger.warning(
                "T1 did not fully settle at the descended grasp pose; "
                "attempting the grasp from the achieved pose."
            )
        self._log_grasp_alignment("grasp")

        if not self._grasp_cube():
            return
        self.valid_grasp = True

        current = self.client.end_effector.get_pose()
        lift_pose = self._offset_z(current, 0.11)
        self._move_cartesian_minimum_jerk(
            lift_pose,
            minimum_duration=1.5,
        )

        target_pos = self._body_pose(self.target_cube_id)[0]
        current = self.client.end_effector.get_pose()
        alignment_pose = Pose(
            [target_pos[0] + 0.055, target_pos[1] + 0.06, current.pos[2]],
            current.ori,
        )
        self._move_cartesian_minimum_jerk(
            alignment_pose,
            minimum_duration=2.0,
        )

        self.skills.level_object_to_table(
            self.cube_id,
            target_pos,
            extra=0.06,
        )
        self._step_simulation(60)

        self.skills.disable_robot_object_collision(self.cube_id)
        self.skills.disable_robot_object_collision(self.target_cube_id)
        self._nail_target_cube()

        stack_welded = self._weld_stack_if_aligned("pre-release placement")
        if not self._release_cube():
            self.valid_grasp = False
            return
        if not stack_welded:
            stack_welded = self._settle_and_weld_stack()
        if not stack_welded:
            self.client.logger.error(
                "T1 cube was released but could not be safely welded."
            )
            self.valid_grasp = False
            return

        current = self.client.end_effector.get_pose()
        retreat_pose = self._offset_z(current, 0.25)
        self._move_cartesian_minimum_jerk(
            retreat_pose,
            minimum_duration=2.0,
        )

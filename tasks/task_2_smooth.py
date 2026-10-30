"""Smooth, contact-aware expert policy for Task 2."""

import math

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from tasks.task_2 import T2Moving


class T2SmoothMoving(T2Moving):
    """Move the bottle using smooth references and contact-aware grasping."""

    SIM_HZ = 240.0
    CONTROL_HZ = 60.0
    PHYSICS_SUBSTEPS = int(SIM_HZ / CONTROL_HZ)

    MAX_LINEAR_SPEED = 0.08
    MAX_LINEAR_ACCELERATION = 0.15
    MAX_ANGULAR_SPEED = 0.8
    MAX_ANGULAR_ACCELERATION = 1.2

    FINGERS = ("thumb", "index", "middle", "ring", "little")
    MIN_FINGER_CONTACTS = 3

    @staticmethod
    def _minimum_jerk(progress):
        return (
            10.0 * progress**3
            - 15.0 * progress**4
            + 6.0 * progress**5
        )

    def _step_control(self):
        """Advance four 240 Hz physics steps per 60 Hz control target."""
        for _ in range(self.PHYSICS_SUBSTEPS):
            self.client.update_simulation(None)

    def _joint_position(self, joint_name):
        handle = self.client.get_joint_handle(joint_name)
        return p.getJointState(
            self.client.robot,
            handle.robot_joint_id,
            physicsClientId=self.client._client,
        )[0]

    def _move_joints_minimum_jerk(self, targets, duration=1.5):
        """Interpolate multiple joints simultaneously."""
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
            self._step_control()

    def _trajectory_duration(
        self,
        start_pose,
        target_pose,
        minimum_duration,
        max_linear_speed,
        max_linear_acceleration,
    ):
        start_pos = np.asarray(start_pose.pos, dtype=float)
        target_pos = np.asarray(target_pose.pos, dtype=float)
        linear_distance = float(np.linalg.norm(target_pos - start_pos))

        start_rotation = R.from_quat(np.asarray(start_pose.ori, dtype=float))
        target_rotation = R.from_quat(np.asarray(target_pose.ori, dtype=float))
        angular_distance = float(
            (target_rotation * start_rotation.inv()).magnitude()
        )

        max_unit_speed = 1.875
        max_unit_acceleration = 10.0 / math.sqrt(3.0)
        return max(
            minimum_duration,
            max_unit_speed * linear_distance / max_linear_speed,
            math.sqrt(
                max_unit_acceleration
                * linear_distance
                / max_linear_acceleration
            ),
            max_unit_speed
            * angular_distance
            / self.MAX_ANGULAR_SPEED,
            math.sqrt(
                max_unit_acceleration
                * angular_distance
                / self.MAX_ANGULAR_ACCELERATION
            ),
        )

    def _move_cartesian_minimum_jerk(
        self,
        target_pose,
        minimum_duration=1.2,
        max_linear_speed=None,
        max_linear_acceleration=None,
        stop_condition=None,
    ):
        """Stream a minimum-jerk pose reference at 60 Hz."""
        max_linear_speed = max_linear_speed or self.MAX_LINEAR_SPEED
        max_linear_acceleration = (
            max_linear_acceleration or self.MAX_LINEAR_ACCELERATION
        )

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
            max_linear_speed,
            max_linear_acceleration,
        )
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))

        for step in range(1, control_steps + 1):
            blend = self._minimum_jerk(step / control_steps)
            position = start_pos + blend * (target_pos - start_pos)
            orientation = orientation_path([blend]).as_quat()[0]
            self.client.move_cartesian(
                Pose(position.tolist(), orientation.tolist()),
                wait=False,
                timeout=None,
                check_collision=False,
            )
            self._step_control()
            if stop_condition is not None and stop_condition():
                return False
        return True

    def _settle_at_pose(self, target_pose, duration=0.5, tolerance=0.008):
        """Give the controller bounded simulated time to remove tracking error."""
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        target = np.asarray(target_pose.pos, dtype=float)
        for _ in range(control_steps):
            actual = np.asarray(
                self.client.end_effector.get_pose().pos,
                dtype=float,
            )
            if np.linalg.norm(target - actual) <= tolerance:
                return True
            self.client.move_cartesian(
                target_pose,
                wait=False,
                timeout=None,
                check_collision=False,
            )
            self._step_control()
        return False

    def _finger_joint_names(self, finger):
        return (
            f"r_hand_{finger}_2_joint",
            f"r_hand_{finger}_3_joint",
        )

    def _hold_finger(self, finger):
        for joint_name in self._finger_joint_names(finger):
            self.client.move_position(
                joint_name,
                self._joint_position(joint_name),
                wait=False,
                check_collision=False,
                velocity=1.0,
                timeout=None,
            )

    def _close_hand_smoothly(self, obj_id, duration=1.5, closure=1.0):
        """Close all fingers together and freeze each on fingertip contact."""
        joints = {}
        starts = {}
        targets = {}
        tip_link_ids = {}
        for finger in self.FINGERS:
            tip = self.client.get_joint_handle(
                f"r_hand_{finger}_3_joint"
            )
            tip_link_ids[finger] = tip.robot_joint_id
            for joint_name in self._finger_joint_names(finger):
                handle = self.client.get_joint_handle(joint_name)
                joints[joint_name] = handle
                starts[joint_name] = self._joint_position(joint_name)
                targets[joint_name] = handle.lower_limit + closure * (
                    handle.upper_limit - handle.lower_limit
                )

        contacted = set()
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))
        for step in range(1, control_steps + 1):
            blend = self._minimum_jerk(step / control_steps)
            for finger in self.FINGERS:
                if finger in contacted:
                    continue
                for joint_name in self._finger_joint_names(finger):
                    target = starts[joint_name] + blend * (
                        targets[joint_name] - starts[joint_name]
                    )
                    self.client.move_position(
                        joint_name,
                        target,
                        wait=False,
                        check_collision=False,
                        velocity=1.5,
                        timeout=None,
                    )

            self._step_control()
            contacts = p.getContactPoints(
                bodyA=self.client.robot,
                bodyB=obj_id,
                physicsClientId=self.client._client,
            )
            contacting_links = {contact[3] for contact in contacts}
            for finger, tip_link_id in tip_link_ids.items():
                if finger not in contacted and tip_link_id in contacting_links:
                    self._hold_finger(finger)
                    contacted.add(finger)
            if len(contacted) == len(self.FINGERS):
                break

        return contacted

    def _open_hand_smoothly(self, duration=1.0):
        targets = {}
        for finger in self.FINGERS:
            for joint_name in self._finger_joint_names(finger):
                handle = self.client.get_joint_handle(joint_name)
                targets[joint_name] = handle.lower_limit
        self._move_joints_minimum_jerk(targets, duration=duration)

    def _grasp_bottle(self):
        """Legacy smooth-task grasp, retained for existing data generation."""
        contacted = self._close_hand_smoothly(
            self.bottle_id,
            duration=1.5,
        )
        if len(contacted) < self.MIN_FINGER_CONTACTS:
            self.client.logger.error(
                f"Only {len(contacted)} fingers contacted the bottle."
            )
            return False

        for _ in range(20):
            self.client.update_simulation(None)
        self.skills.secure_grasp(self.bottle_id)
        if not self.skills.active_constraints:
            self.client.logger.error("No grasp constraints were created.")
            return False
        return True

    def _release_bottle(self):
        """Legacy smooth-task release, retained for existing recordings."""
        self.skills.release_grasp()
        for _ in range(120):
            self.client.update_simulation(None)
        self._open_hand_smoothly(duration=1.0)
        return True

    def _support_contact(self):
        """Detect bottle contact with a body other than the robot/computer."""
        contacts = p.getContactPoints(
            bodyA=self.bottle_id,
            physicsClientId=self.client._client,
        )
        excluded = {self.client.robot, self.computer_id}
        return any(contact[2] not in excluded for contact in contacts)

    def _right_hand_touches_bottle(self):
        """Return whether a right-hand link is touching the bottle."""
        hand_links = {self.client.end_effector.link_id}
        for joint in self.client.joints:
            if joint.name.startswith("r_hand_"):
                hand_links.add(joint.robot_joint_id)

        contacts = p.getContactPoints(
            bodyA=self.client.robot,
            bodyB=self.bottle_id,
            physicsClientId=self.client._client,
        )
        return any(contact[3] in hand_links for contact in contacts)

    def _hand_pose_for_bottle_pose(self, bottle_target_pos, bottle_target_orn):
        """Preserve the measured bottle-to-hand transform."""
        bottle_pos, bottle_orn = self.client.getBasePositionAndOrientation(
            self.bottle_id
        )
        hand_pose = self.client.end_effector.get_pose()
        inv_bottle_pos, inv_bottle_orn = p.invertTransform(
            bottle_pos,
            bottle_orn,
        )
        local_hand_pos, local_hand_orn = p.multiplyTransforms(
            inv_bottle_pos,
            inv_bottle_orn,
            hand_pose.pos,
            hand_pose.ori,
        )
        hand_target_pos, hand_target_orn = p.multiplyTransforms(
            bottle_target_pos,
            bottle_target_orn,
            local_hand_pos,
            local_hand_orn,
        )
        return Pose(hand_target_pos, hand_target_orn)

    def execute_expert(self):
        self.valid_grasp = False
        self.skills.set_dynamics(self.bottle_id, mass=0.2)
        self.skills.set_dynamics(self.computer_id, mass=0.2)
        self.skills.set_object_to_track(self.bottle_id)

        # Smoothly move the unused arm away and pre-position the thumb.
        self._move_joints_minimum_jerk(
            {
                "l_shoulder_pitch": -1.6,
                "r_hand_thumb_0_joint": 1.209,
            },
            duration=1.5,
        )
        self.skills.track_target_with_head()

        bottle_pos = self.client.getBasePositionAndOrientation(
            self.bottle_id
        )[0]
        grasp_position = np.array(
            [
                bottle_pos[0] + 0.03,
                bottle_pos[1] + 0.005,
                bottle_pos[2] + 0.06,
            ],
            dtype=float,
        )
        grasp_orientation = [
            -0.02655094675719738,
            -0.10425235331058502,
            -0.4081196188926697,
            0.9065676331520081,
        ]

        # Stop in free space first, then approach slowly until either the
        # requested pose or physical hand/bottle contact is reached.
        current_pose = self.client.end_effector.get_pose()
        approach_direction = (
            np.asarray(current_pose.pos, dtype=float) - grasp_position
        )
        approach_norm = float(np.linalg.norm(approach_direction))
        if approach_norm > 1e-6:
            approach_direction /= approach_norm
        else:
            approach_direction = np.array([1.0, 0.0, 0.0])

        pregrasp_position = grasp_position + 0.05 * approach_direction
        pregrasp_pose = Pose(
            pregrasp_position.tolist(),
            grasp_orientation,
        )
        grasp_pose = Pose(
            grasp_position.tolist(),
            grasp_orientation,
        )

        self._move_cartesian_minimum_jerk(
            pregrasp_pose,
            minimum_duration=2.0,
        )
        reached_grasp_pose = self._move_cartesian_minimum_jerk(
            grasp_pose,
            minimum_duration=1.0,
            max_linear_speed=0.02,
            max_linear_acceleration=0.04,
            stop_condition=self._right_hand_touches_bottle,
        )

        actual_position = np.asarray(
            self.client.end_effector.get_pose().pos,
            dtype=float,
        )
        grasp_error = float(np.linalg.norm(actual_position - grasp_position))
        contact_terminated_approach = not reached_grasp_pose
        if (
            not contact_terminated_approach
            and not self._right_hand_touches_bottle()
            and grasp_error > 0.02
        ):
            self.client.logger.error(
                f"Could not reach the bottle: error={grasp_error:.3f} m."
            )
            return

        if not self._grasp_bottle():
            return
        self.valid_grasp = True

        current = self.client.end_effector.get_pose()
        lift_pose = Pose(
            [current.pos[0], current.pos[1], current.pos[2] + 0.09],
            current.ori,
        )
        self._move_cartesian_minimum_jerk(
            lift_pose,
            minimum_duration=1.5,
        )

        self._move_joints_minimum_jerk(
            {
                "r_shoulder_pitch": -0.891,
                "r_shoulder_roll": 1.241,
                "r_shoulder_yaw": 1.054,
                "r_elbow": 1.846,
                "r_wrist_prosup": -0.959,
                "r_wrist_yaw": 0.005,
            },
            duration=1.5,
        )

        computer_pos = np.asarray(
            self.client.getBasePositionAndOrientation(self.computer_id)[0],
            dtype=float,
        )
        self.target_final_pos = np.array(
            [computer_pos[0] + 0.25, computer_pos[1], computer_pos[2]],
            dtype=float,
        )

        bottle_pos, bottle_orn = self.client.getBasePositionAndOrientation(
            self.bottle_id
        )
        transport_bottle_pos = [
            self.target_final_pos[0],
            self.target_final_pos[1],
            bottle_pos[2],
        ]
        transport_pose = self._hand_pose_for_bottle_pose(
            transport_bottle_pos,
            bottle_orn,
        )
        self._move_cartesian_minimum_jerk(
            transport_pose,
            minimum_duration=2.5,
        )

        bottle_pos, bottle_orn = self.client.getBasePositionAndOrientation(
            self.bottle_id
        )
        bottle_yaw = p.getEulerFromQuaternion(bottle_orn)[2]
        flat_bottle_orn = p.getQuaternionFromEuler([0.0, 0.0, bottle_yaw])
        level_pose = self._hand_pose_for_bottle_pose(
            bottle_pos,
            flat_bottle_orn,
        )
        self._move_cartesian_minimum_jerk(
            level_pose,
            minimum_duration=1.5,
            max_linear_speed=0.05,
            max_linear_acceleration=0.10,
        )

        bottle_pos, bottle_orn = self.client.getBasePositionAndOrientation(
            self.bottle_id
        )
        placement_bottle_pos = [
            self.target_final_pos[0],
            self.target_final_pos[1],
            self.target_final_pos[2],
        ]
        placement_pose = self._hand_pose_for_bottle_pose(
            placement_bottle_pos,
            bottle_orn,
        )
        self._move_cartesian_minimum_jerk(
            placement_pose,
            minimum_duration=1.5,
            max_linear_speed=0.03,
            max_linear_acceleration=0.06,
            stop_condition=self._support_contact,
        )

        if not self._release_bottle():
            self.valid_grasp = False
            return

        current = self.client.end_effector.get_pose()
        retreat_pose = Pose(
            [current.pos[0], current.pos[1], current.pos[2] + 0.11],
            current.ori,
        )
        self._move_cartesian_minimum_jerk(
            retreat_pose,
            minimum_duration=1.5,
        )

    def check_success(self):
        if not self.valid_grasp:
            return False
        bottle_pos = np.asarray(
            self.client.getBasePositionAndOrientation(self.bottle_id)[0],
            dtype=float,
        )
        distance_xy = float(
            np.linalg.norm(bottle_pos[:2] - self.target_final_pos[:2])
        )
        if distance_xy > 0.05:
            self.client.logger.error(
                f"Bottle is {distance_xy:.3f} m from its target."
            )
            return False
        return True

"""Task 3 with time-parameterized minimum-jerk Cartesian motion.

PyBullet is stepped at 240 Hz while the data collector samples at 5 Hz. Each
Cartesian reference in this task is generated once per physics step, so the
collected end-effector deltas sample the same smooth reference.
"""

import math

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from tasks.task_3 import T3Pushing


class T3SmoothPushing(T3Pushing):
    """Push the green box using velocity- and acceleration-limited paths."""

    SIM_HZ = 240.0
    DATA_HZ = 5.0
    
    CONTROL_HZ = 60.0
    PHYSICS_SUBSTEPS = int(SIM_HZ / CONTROL_HZ)

    # Reference-trajectory limits. Durations are increased automatically to
    # satisfy both limits; tune these rather than choosing waypoint counts.
    MAX_LINEAR_SPEED = 0.12  # m/s
    MAX_LINEAR_ACCELERATION = 0.25  # m/s^2
    MAX_ANGULAR_SPEED = 1.2  # rad/s
    MAX_ANGULAR_ACCELERATION = 2 # rad/s^2

    # Ten samples at 5 Hz give each major movement a useful action profile.
    MIN_SAMPLES_PER_SEGMENT = 10

    def setup_environment(self):
        super().setup_environment()

        # BaseTask starts the collector after setup and randomization. Keep
        # this posture change outside the recorded Cartesian demonstration.
        self.skills.set_object_to_track(self.green_box_id)
        self.skills.set_dynamics(
            self.green_box_id, mass=0.5, lat_fric=0.5
        )
        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)
        self.skills.open_hand()

    @staticmethod
    def _minimum_jerk(progress):
        """Quintic time scaling with zero endpoint velocity/acceleration."""
        return (
            10.0 * progress**3
            - 15.0 * progress**4
            + 6.0 * progress**5
        )

    def _trajectory_duration(self, start_pose, target_pose):
        """Return a duration satisfying the configured reference limits."""
        start_pos = np.asarray(start_pose.pos, dtype=float)
        target_pos = np.asarray(target_pose.pos, dtype=float)
        linear_distance = float(np.linalg.norm(target_pos - start_pos))

        start_rotation = R.from_quat(
            np.asarray(start_pose.ori, dtype=float)
        )
        target_rotation = R.from_quat(
            np.asarray(target_pose.ori, dtype=float)
        )
        angular_distance = float(
            (target_rotation * start_rotation.inv()).magnitude()
        )

        # For s(u)=10u^3-15u^4+6u^5:
        # max(ds/du)=1.875 and max(abs(d2s/du2))=10/sqrt(3).
        max_unit_speed = 1.875
        max_unit_acceleration = 10.0 / math.sqrt(3.0)
        minimum_duration = (
            self.MIN_SAMPLES_PER_SEGMENT / self.DATA_HZ
        )

        durations = [
            minimum_duration,
            max_unit_speed
            * linear_distance
            / self.MAX_LINEAR_SPEED,
            math.sqrt(
                max_unit_acceleration
                * linear_distance
                / self.MAX_LINEAR_ACCELERATION
            ),
            max_unit_speed
            * angular_distance
            / self.MAX_ANGULAR_SPEED,
            math.sqrt(
                max_unit_acceleration
                * angular_distance
                / self.MAX_ANGULAR_ACCELERATION
            ),
        ]
        return max(durations)

    def _move_cartesian_minimum_jerk(self, target_pose):
        """Issue one smooth Cartesian reference per 240 Hz physics step."""
        start_pose = self.client.end_effector.get_pose()
        start_pos = np.asarray(start_pose.pos, dtype=float)
        target_pos = np.asarray(target_pose.pos, dtype=float)

        start_quaternion = np.asarray(start_pose.ori, dtype=float)
        target_quaternion = np.asarray(target_pose.ori, dtype=float)
        rotations = R.from_quat(
            np.stack([start_quaternion, target_quaternion])
        )
        orientation_path = Slerp([0.0, 1.0], rotations)

        duration = self._trajectory_duration(start_pose, target_pose)
        #physics_steps = max(1, int(math.ceil(duration * self.SIM_HZ)))
        #step_timeout = 1.0 / self.SIM_HZ
        #step_timeout = 0.25  
        
        control_steps = max(1, int(math.ceil(duration * self.CONTROL_HZ)))

        for step in range(1, control_steps + 1):
            progress = step / control_steps
            blend = self._minimum_jerk(progress)
            position = start_pos + blend * (target_pos - start_pos)
            orientation = orientation_path([blend]).as_quat()[0]

            self.client.move_cartesian(
                Pose(position.tolist(), orientation.tolist()),
                wait=False,
                timeout=None,
                check_collision=False,
            )
            
            for _ in range(self.PHYSICS_SUBSTEPS):
                self.client.update_simulation(None)

    def execute_expert(self):
        (
            face_normal,
            face_name,
            push_orientation,
            box_pos,
            box_xy,
        ) = self.skills.get_push_alignment(
            self.green_box_id, self.target_final_xy
        )
        print(
            f"Aligning to push the {face_name} face with a "
            "minimum-jerk trajectory."
        )

        offset_distance = 0.08
        start_xy = box_xy + face_normal * offset_distance
        box_min, box_max = p.getAABB(
            self.green_box_id,
            physicsClientId=self.client._client,
        )

        # Push near the upper part of the box while remaining below its top.
        push_z = box_max[2] - 0.02
        pre_push_z = push_z + 0.1
        eef_offset_x = 0.07

        pre_push_pose = Pose(
            [
                start_xy[0] + eef_offset_x,
                start_xy[1],
                pre_push_z,
            ],
            push_orientation,
        )
        start_pose = Pose(
            [
                start_xy[0] + eef_offset_x,
                start_xy[1],
                push_z,
            ],
            push_orientation,
        )
        end_xy = self.target_final_xy + face_normal * offset_distance

        end_pose = Pose(
            [
                end_xy[0] + eef_offset_x,
                end_xy[1],
                push_z,
            ],
            push_orientation,
        )
        retreat_pose = Pose(
            [
                end_xy[0] + eef_offset_x,
                end_xy[1],
                pre_push_z,
            ],
            push_orientation,
        )

        self._move_cartesian_minimum_jerk(pre_push_pose)
        self._move_cartesian_minimum_jerk(start_pose)
        self._move_cartesian_minimum_jerk(end_pose)
        self._move_cartesian_minimum_jerk(retreat_pose)

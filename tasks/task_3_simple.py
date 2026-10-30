"""Simplified T3 pushing task.

The end effector is placed at the correct pushing height and orientation before
the collector starts. Consequently, the recorded expert trajectory contains
only motion in the XY plane.
"""

import numpy as np
import pybullet as p
from icub_pybullet.utils import Pose

from tasks.base_task import BaseTask


class T3SimplePushing(BaseTask):
    CONFIG_PATH = "configs/t3_simple.yaml"
    BASE_PROMPT = "Push the green box to the centre of the table."
    OBJECT_NAMES = ["box", "other_box"]

    def setup_environment(self):
        self.red_box_id = self.client.get_object_id("box")
        self.green_box_id = self.client.get_object_id("other_box")
        
        min_x, min_y = (-0.6, -0.4)
        max_x, max_y = (-0.22, 0.339)
        self.target_final_xy = np.array(
            [(min_x + max_x) / 2, (min_y + max_y) / 2]
        )

        self.skills.set_object_to_track(self.green_box_id)
        self.skills.set_dynamics(
            self.green_box_id, mass=0.5, lat_fric=0.5
        )

        # BaseTask.run() calls setup_environment() and object_randomizer()
        # before DataCollector.start(). Prepare here only when randomization
        # is disabled; otherwise prepare after the box has been randomized.
        if not self.randomizer:
            self._prepare_initial_ee_pose()

    def object_randomizer(self):
        if getattr(self.client.config.camera, "randomize_camera", False):
            base_pos = getattr(
                self.client.config.camera,
                "fixed_camera_pos",
                [-0.6742, -0.0194, 0.4362],
            )
            base_target = getattr(
                self.client.config.camera,
                "fixed_camera_target",
                [-0.2265, -0.0107, -0.4580],
            )
            pos_noise = getattr(
                self.client.config.camera, "random_pos_range", 0.05
            )
            target_noise = getattr(
                self.client.config.camera, "random_target_range", 0.02
            )

            new_pos = [
                base_pos[i] + np.random.uniform(-pos_noise, pos_noise)
                for i in range(3)
            ]
            new_target = [
                base_target[0]
                + np.random.uniform(-target_noise, target_noise),
                base_target[1]
                + np.random.uniform(-target_noise, target_noise),
                base_target[2],
            ]
            self.client.config.camera.fixed_camera_pos = new_pos
            self.client.config.camera.fixed_camera_target = new_target

            if (
                hasattr(self.client, "visualizer")
                and "fixed_camera" in self.client.visualizer.eye_windows
            ):
                camera = self.client.visualizer.eye_windows["fixed_camera"]
                camera.set_camera_pose(new_pos, new_target)
                camera.update_view()

        aabb_min, aabb_max = p.getAABB(
            self.green_box_id, physicsClientId=self.client._client
        )
        random_x = np.random.uniform(aabb_min[0], aabb_max[0] - 0.1)
        random_y = np.random.uniform(aabb_min[1], aabb_max[1])
        self.client.resetBasePositionAndOrientation(
            self.green_box_id,
            [random_x, random_y, aabb_min[2]],
            [0, 0, 1, 1],
        )

        red_box_pos = self.client.getBasePositionAndOrientation(
            self.red_box_id
        )[0]
        if np.random.randint(0, 2) == 0:
            safety_margin = 0.04
            random_x = np.random.uniform(
                -0.59 + safety_margin, -0.19 - safety_margin
            )
            new_red_pos = [random_x, red_box_pos[1], red_box_pos[2]]
        else:
            safety_margin = 0.05
            random_y = np.random.uniform(
                -0.39 + safety_margin, -0.001 - safety_margin
            )
            new_red_pos = [red_box_pos[0], random_y, red_box_pos[2]]

        self.client.resetBasePositionAndOrientation(
            self.red_box_id, new_red_pos, [0, 0, 1, 1]
        )

        # This still runs before collector.start(), so repositioning the hand
        # in Z and changing its orientation do not enter the demonstration.
        self._prepare_initial_ee_pose()

    def _prepare_initial_ee_pose(self):
        self.skills.move_joint("l_shoulder_pitch", -1.6, wait=True)
        self.skills.open_hand()

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
            f"Pre-aligning the end effector with the {face_name} face "
            "before recording."
        )

        start_xy = box_xy + face_normal * 0.08
        self.push_z = box_pos[2] + 0.04
        self.eef_offset_x = 0.07
        self.push_orientation = push_orientation

        initial_pose = Pose(
            [
                start_xy[0] + self.eef_offset_x,
                start_xy[1],
                self.push_z,
            ],
            self.push_orientation,
        )
        self.client.move_cartesian(
            initial_pose, timeout=3.0, check_collision=False
        )

        while not self.client.motion_done(check_collision=False):
            self.client.update_simulation()
            self.skills.track_target_with_head()

    def execute_expert(self):
        # Z and orientation remain constant for every recorded waypoint.

        end_pose = Pose(
            [
                self.target_final_xy[0] + self.eef_offset_x,
                self.target_final_xy[1] + 0.05,
                self.push_z,
            ],
            self.push_orientation,
        )
        self.skills.move_cartesian_smoothly(
            end_pose, steps=50, timeout=2.0
        )

    def check_success(self) -> bool:
        red_min, red_max = p.getAABB(
            self.red_box_id,
            physicsClientId=self.client._client,
        )
        green_min, green_max = p.getAABB(
            self.green_box_id,
            physicsClientId=self.client._client,
        )

        aabbs_overlap = all(
            red_min[axis] < green_max[axis]
            and red_max[axis] > green_min[axis]
            for axis in range(3)
        )
        if aabbs_overlap:
            self.client.logger.error(
                "Red and green box AABBs overlap."
            )
            return False

        box_pos = self.client.getBasePositionAndOrientation(
            self.green_box_id
        )[0]
        distance = np.linalg.norm(
            np.asarray(box_pos[:2]) - self.target_final_xy
        )
        if distance > 0.1:
            self.client.logger.error(
                f"Box is {distance:.3f}m away from the center."
            )
            return False
        return True

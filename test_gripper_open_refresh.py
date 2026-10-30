import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from core.robot_skills import RobotSkills, GripperState

class OpenRefreshTests(unittest.TestCase):
    def test_final_target_is_reissued_until_joint_arrives(self):
        s = RobotSkills(Mock(robot=1, _client=0))
        s.gripper_state = GripperState.OPENING
        s.gripper_opening_step = s.gripper_opening_steps = 60
        s.gripper_opening_fingers = ("index",)
        s.gripper_finger_joint_names = {"index": ["joint"]}
        s.gripper_joint_handles = {"joint": SimpleNamespace(robot_joint_id=2)}
        s.gripper_open_targets = {"joint": 0.0}
        s.gripper_opening_starts = {"joint": 1.0}
        with patch("core.robot_skills.p") as physics:
            physics.getJointState.return_value = (.1, 0, 0, 0)
            result = s.execute_gripper_post_action(1)
            self.assertEqual(result["state_after"], "OPENING")
            self.assertAlmostEqual(result["open_joint_errors_rad"]["joint"], .1)
            s.client.move_position.assert_called_with(
                "joint", 0.0, wait=False, check_collision=False,
                velocity=s.gripper_opening_velocity, timeout=1.0)
            physics.getJointState.return_value = (.01, 0, 0, 0)
            result = s.execute_gripper_post_action(1)
            self.assertEqual(result["state_after"], "OPEN")
            self.assertEqual(s.gripper_opening_step, 60)
            self.assertEqual(s.client.move_position.call_count, 2)

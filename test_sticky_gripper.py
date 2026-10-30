"""Controller tests with mocked physics: no simulator connection or stepping."""
import unittest
from unittest.mock import Mock, patch

from core.robot_skills import RobotSkills, GripperState


class StickyGripperTest(unittest.TestCase):
    def setUp(self):
        self.physics = patch("core.robot_skills.p").start()
        self.addCleanup(patch.stopall)
        pose = ([0, 0, 0], [0, 0, 0, 1])
        self.physics.getLinkState.return_value = pose
        self.physics.getBasePositionAndOrientation.return_value = pose
        self.physics.invertTransform.return_value = pose
        self.physics.multiplyTransforms.return_value = pose
        self.physics.createConstraint.side_effect = range(100, 120)
        self.skills = s = RobotSkills(Mock(robot=1, _client=0))
        s.gripper_attachment_policy = "per_finger_then_validate"
        s.gripper_state = GripperState.CLOSING
        s.gripper_closing_strategy = "sequential"
        s.gripper_current_finger = "thumb"
        s.gripper_closing_fingers = ("thumb", "index", "middle", "ring")
        s.gripper_closing_order = s.gripper_closing_fingers
        s.gripper_mandatory_fingers = {"thumb"}
        s.gripper_support_fingers = {"index", "middle", "ring"}
        s.gripper_minimum_support_fingers = 2
        s.gripper_post_close_settle_steps = 30
        s.gripper_finger_tip_link_ids = dict(zip(s.gripper_closing_fingers, range(4)))
        s._hold_gripper_finger = Mock()
        s._advance_gripper_closing = Mock()
        s._gripper_closing_finished = Mock(return_value=False)
        s._find_grasp_candidate = Mock(return_value=(7, {"index"}))

    def test_contact_attaches_non_active_finger_once_without_success(self):
        s = self.skills
        s.execute_gripper_post_action(0)
        s.execute_gripper_post_action(0)
        self.assertEqual(s.gripper_frozen_fingers, {"index"})
        self.physics.createConstraint.assert_called_once()
        s._hold_gripper_finger.assert_called_once_with("index")
        self.assertEqual(s.gripper_state, GripperState.CLOSING)
        self.assertIsNone(s.gripper_held_object_id)

    def test_success_requires_finished_closure_and_three_live_checks(self):
        s = self.skills
        s._find_grasp_candidate.return_value = (7, {"thumb", "index", "middle"})
        s.execute_gripper_post_action(0)
        self.assertEqual(s.gripper_contact_steps, 0)
        s._gripper_closing_finished.return_value = True
        for _ in range(2):
            s.execute_gripper_post_action(0)
            self.assertEqual(s.gripper_state, GripperState.CLOSING)
        s.execute_gripper_post_action(0)
        self.assertEqual(s.gripper_state, GripperState.HOLDING)
        self.assertEqual(s.gripper_held_object_id, 7)
        self.assertEqual(self.physics.createConstraint.call_count, 3)

    def test_attached_fingers_do_not_replace_live_contacts(self):
        s = self.skills
        s._find_grasp_candidate.return_value = (7, {"thumb", "index", "middle"})
        s.execute_gripper_post_action(0)
        s._gripper_closing_finished.return_value = True
        s.execute_gripper_post_action(0)
        s._find_grasp_candidate.return_value = (7, {"index", "middle", "ring"})
        s.execute_gripper_post_action(0)
        self.assertEqual(s.gripper_contact_steps, 0)
        self.assertEqual(s.gripper_state, GripperState.CLOSING)

    def test_failed_grasp_removes_temporary_constraints(self):
        s = self.skills
        s.execute_gripper_post_action(0)
        s._gripper_closing_finished.return_value = True
        s._find_grasp_candidate.return_value = (None, set())
        for _ in range(30):
            s.execute_gripper_post_action(0)
        self.assertEqual(s.gripper_state, GripperState.CLOSED_EMPTY)
        self.assertEqual(s.gripper_constraint_ids, [])
        self.assertEqual(s.active_constraints, [])
        self.physics.removeConstraint.assert_called_once_with(100, physicsClientId=0)

    def test_open_releases_temporary_constraints(self):
        s = self.skills
        s.execute_gripper_post_action(0)
        s._advance_gripper_opening = Mock()
        s.execute_gripper_pre_action(1)
        self.assertEqual(s.gripper_state, GripperState.OPENING)
        self.assertEqual(s.gripper_constraint_ids, [])
        self.physics.removeConstraint.assert_called_once_with(100, physicsClientId=0)

    def test_candidate_stays_locked_to_first_cube(self):
        s = self.skills
        s.gripper_candidate_object_ids = [7, 8]
        s.gripper_contact_candidate = 7
        self.physics.getContactPoints.return_value = []
        self.assertEqual(RobotSkills._find_grasp_candidate(s), (None, set()))
        self.physics.getContactPoints.assert_called_once_with(
            bodyA=1, bodyB=7, physicsClientId=0,
        )

    def test_profile_retains_original_success_requirements(self):
        old = RobotSkills.GRIPPER_CONTROLLER_PROFILES["t1_new_smooth"]
        new = RobotSkills.GRIPPER_CONTROLLER_PROFILES["t1_sticky_fingertips"]
        self.assertEqual(old["attachment_policy"], "atomic_when_valid")
        self.assertEqual(old["contact_hold_policy"], "closed_target")
        for key in ("required_fingers", "mandatory_fingers", "support_fingers",
                    "minimum_support_fingers", "contact_confirmation_steps",
                    "closing_strategy", "finger_closing_duration"):
            self.assertEqual(new[key], old[key])


if __name__ == "__main__":
    unittest.main()

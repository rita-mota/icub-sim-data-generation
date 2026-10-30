"""Smooth Task 2 demonstrations using the shared binary gripper controller."""

import math

from core.robot_skills import GripperState
from tasks.task_2_smooth import T2SmoothMoving


class T2NewSmoothMoving(T2SmoothMoving):
    """Collect T2 demonstrations with the inference-time hand controller."""

    GRIPPER_PROFILE = "t2_new_smooth"

    def _run_gripper_command(self, command, timeout):
        """Advance the non-blocking controller at its native 60 Hz rate."""
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
            self._step_control()
            diagnostics = self.skills.execute_gripper_post_action(command)
            if self.skills.gripper_state in terminal_states:
                break

        return self.skills.gripper_state, diagnostics

    def _grasp_bottle(self):
        initial_state = self.skills.initialize_gripper_controller(
            candidate_object_ids=[self.bottle_id],
            hand="right",
            profile=self.GRIPPER_PROFILE,
            initial_command=1,
        )
        if initial_state != GripperState.OPEN:
            self.client.logger.error(
                "Shared gripper controller did not initialize with an open hand: "
                f"{initial_state.name}."
            )
            return False

        final_state, diagnostics = self._run_gripper_command(
            command=0,
            timeout=self.skills.gripper_closing_duration + 1.0,
        )
        if final_state != GripperState.HOLDING:
            self.client.logger.error(
                "Shared gripper controller failed to grasp the bottle: "
                f"state={final_state.name}, diagnostics={diagnostics}."
            )
            return False
        return True

    def _release_bottle(self):
        final_state, diagnostics = self._run_gripper_command(
            command=1,
            timeout=self.skills.gripper_opening_duration + 1.0,
        )
        if final_state != GripperState.OPEN:
            self.client.logger.error(
                "Shared gripper controller failed to open the hand: "
                f"state={final_state.name}, diagnostics={diagnostics}."
            )
            return False
        return True

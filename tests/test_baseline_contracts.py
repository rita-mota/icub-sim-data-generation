"""Lightweight behavioral contracts captured before publication refactoring."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_PARENT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from sim_data_collection.core.data_collector import DataCollector
from sim_data_collection.core.robot_skills import RobotSkills
from sim_data_collection.core.spill import sample_spill
from sim_data_collection.tasks.task_1_smooth import T1SmoothStacking
from sim_data_collection.tasks.task_2_smooth import T2SmoothMoving
from sim_data_collection.tasks.task_3_smooth import T3SmoothPushing
from sim_data_collection.tasks.task_4_smooth import T4SmoothOpening
from sim_data_collection.tasks.task_5_smooth import T5SmoothCleaning


EXPECTED_HASHES = {
    "configs/t1.yaml": "1572dd148f536cadb2bc30bed7425e9428f15189804f080489ef94cfa569b2ca",
    "configs/t2.yaml": "3a39e66e93cb513da9c53bcc15f1f21f65504c7a3673bd60a5708048a9a1849d",
    "configs/t3.yaml": "6b6883d8c13c1a4e13a7975ba85c10df5ddafd642e3833f68e922ae4a2b4237f",
    "configs/t3_simple.yaml": "af92d7007d0f3f1d1b672ba29a1cea4e3a3f6e6136e3774cd6fa1874f805d40c",
    "configs/t4.yaml": "58139ce14d9ae17953bec59013c02329da731e4a5fce6e6b3ddf2bee1fbcdf5f",
    "configs/t5.yaml": "129ac719e30bdaa5d90a49e5947c8a5774d9644bdda786cec4134695ee81ef44",
    "recordings/t1/t1_prompts.txt": "19d37a8daa6920865cda64fcf568bbeeb86f2f67dd07df08f242b79d9515e3f6",
    "recordings/t2/t2_prompts.txt": "5fda34545d00b45780beca5db45e75b487035af2dc373eba337c6ced23f7db4b",
    "recordings/t3/t3_prompts.txt": "d6407a0e1c806d638ef1a020e32def26dbf8691cf57e38f3bede260899fe2b4a",
    "recordings/t4/t4_prompts.txt": "3077744c6d4e8ec241e0dc23db68c1f76827ffae32806ae4108b86b36f76e1b4",
    "recordings/t5/t5_prompts.txt": "d70a51eb2831aaaf71ce1a72d4681310781637189f5266a8fdb691bfd8f1fd55",
}

EXPECTED_TASK_REGISTRY = {
    "t1": "T1Stacking",
    "t1_smooth": "T1SmoothStacking",
    "t2": "T2Moving",
    "t2_smooth": "T2SmoothMoving",
    "t2_new_smooth": "T2NewSmoothMoving",
    "t3": "T3Pushing",
    "t3_simple": "T3SimplePushing",
    "t3_smooth": "T3SmoothPushing",
    "t4": "T4Opening",
    "t4_smooth": "T4SmoothOpening",
    "t5": "T5Cleaning",
    "t5_smooth": "T5SmoothCleaning",
}

PUBLIC_SMOOTH_TASKS = {
    "t1_smooth": "T1SmoothStacking",
    "t2_new_smooth": "T2NewSmoothMoving",
    "t3_smooth": "T3SmoothPushing",
    "t4_smooth": "T4SmoothOpening",
    "t5_smooth": "T5SmoothCleaning",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_task_registry() -> dict[str, str]:
    tree = ast.parse((ROOT / "main_collect.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_task_class":
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "registry"
                        for target in statement.targets
                    )
                    and isinstance(statement.value, ast.Dict)
                ):
                    return {
                        ast.literal_eval(key): value.id
                        for key, value in zip(statement.value.keys, statement.value.values)
                        if isinstance(value, ast.Name)
                    }
    raise AssertionError("Could not locate get_task_class() registry")


class StaticScientificContractTests(unittest.TestCase):
    def test_configuration_and_prompt_hashes(self):
        actual = {relative: sha256(ROOT / relative) for relative in EXPECTED_HASHES}
        self.assertEqual(actual, EXPECTED_HASHES)

    def test_task_registry_is_unchanged(self):
        self.assertEqual(parse_task_registry(), EXPECTED_TASK_REGISTRY)

    def test_public_smooth_task_selection_is_available(self):
        registry = parse_task_registry()
        self.assertEqual(
            {name: registry[name] for name in PUBLIC_SMOOTH_TASKS},
            PUBLIC_SMOOTH_TASKS,
        )

    def test_configs_retain_core_simulator_contract(self):
        for config_name in ("t1.yaml", "t2.yaml", "t3.yaml", "t4.yaml", "t5.yaml"):
            text = (ROOT / "configs" / config_name).read_text(encoding="utf-8")
            self.assertIn('robot_urdf_path: "iCub/full.urdf"', text)
            self.assertIn('end_effector: "r_hand"', text)
            self.assertIn("simulation_step: 240", text)


class TrajectoryAndRandomizationContractTests(unittest.TestCase):
    def test_minimum_jerk_reference_values_match_all_smooth_tasks(self):
        progress = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = np.array([0.0, 0.103515625, 0.5, 0.896484375, 1.0])
        helpers = (
            T1SmoothStacking._minimum_jerk,
            T2SmoothMoving._minimum_jerk,
            T3SmoothPushing._minimum_jerk,
            T4SmoothOpening._minimum_jerk,
            T5SmoothCleaning._minimum_jerk,
        )
        for helper in helpers:
            with self.subTest(helper=helper.__qualname__):
                np.testing.assert_allclose(
                    [helper(value) for value in progress],
                    expected, rtol=0.0, atol=1e-15
                )

    def test_seeded_spill_is_repeatable_and_matches_reference(self):
        first = sample_spill(seed=12345)
        second = sample_spill(seed=12345)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(first["spill_0"]["deleted_step"], -1)
        np.testing.assert_allclose(
            first["spill_0"]["pos"],
            [-0.32731930112190455, 0.06594493232044898, -0.1210110063463354],
            rtol=0.0,
            atol=1e-15,
        )
        self.assertAlmostEqual(first["spill_0"]["radius"], 0.023070158628567802)


class GripperProfileContractTests(unittest.TestCase):
    def test_supported_profile_constants_match_reference(self):
        profiles = RobotSkills.GRIPPER_CONTROLLER_PROFILES
        expected = {
            "t1_new_smooth": {
                "attachment_policy": "per_finger_immediate",
                "closing_strategy": "sequential",
                "closing_duration": 1.6,
                "finger_closing_duration": 0.4,
                "mandatory_fingers": ("thumb",),
                "minimum_support_fingers": 2,
                "required_fingers": 2,
                "thumb_opposition_target": 1.209,
            },
            "t2_new_smooth": {
                "attachment_policy": "per_finger_immediate",
                "closing_strategy": "simultaneous",
                "closing_duration": 1.5,
                "required_fingers": 3,
                "thumb_opposition_target": 1.209,
            },
            "t4_new_smooth": {
                "attachment_policy": "single_hand_atomic",
                "closing_strategy": "simultaneous",
                "closing_duration": 1.5,
                "contact_confirmation_steps": 30,
                "contact_link_policy": "controlled_phalanges",
                "required_fingers": 2,
            },
        }
        for profile_name, contract in expected.items():
            with self.subTest(profile=profile_name):
                profile = profiles[profile_name]
                for key, value in contract.items():
                    self.assertEqual(profile[key], value)

    def test_gripper_command_hysteresis(self):
        skills = RobotSkills(Mock(robot=1, _client=0))
        self.assertEqual(skills._decode_gripper_command(0.9), 1)
        self.assertEqual(skills._decode_gripper_command(0.1), 0)
        self.assertEqual(skills._decode_gripper_command(0.5), 1)
        skills.last_gripper_command = 0
        self.assertEqual(skills._decode_gripper_command(0.5), 0)
        skills.last_gripper_command = 1
        self.assertEqual(skills._decode_gripper_command(0.5), 1)


class DatasetOutputContractTests(unittest.TestCase):
    def test_small_aligned_dataset_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DataCollector(
                client=object(),
                camera_keys=[],
                save_dir=directory,
                hz=5.0,
                object_ids=[],
            )
            collector.success = True
            collector._timestamps = [0.2, 0.4]
            collector._proprio = [
                np.array([1.0, 2.0], dtype=np.float32),
                np.array([1.5, 2.5], dtype=np.float32),
            ]
            collector._ee_poses = [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            ]
            collector._ee_deltas = [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            ]
            collector._obj_poses = [
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            ]
            collector._joint_names = ["joint_a", "joint_b"]
            collector.cam_pos = [-0.5, 0.0, 0.4]
            collector.cam_target = [-0.2, 0.0, -0.4]
            collector.cam_up = [0.0, 0.0, 1.0]

            collector.save()

            output = Path(directory)
            expected_files = {
                "timestamps.npy",
                "proprio.npy",
                "ee_poses.npy",
                "ee_deltas.npy",
                "obj_poses.npy",
                "joint_names.txt",
                "ee_columns.txt",
                "camera_info.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected_files)

            timestamps = np.load(output / "timestamps.npy")
            proprio = np.load(output / "proprio.npy")
            ee_poses = np.load(output / "ee_poses.npy")
            ee_deltas = np.load(output / "ee_deltas.npy")
            obj_poses = np.load(output / "obj_poses.npy")

            self.assertEqual(timestamps.shape, (2,))
            self.assertEqual(proprio.shape, (2, 2))
            self.assertEqual(ee_poses.shape, (2, 7))
            self.assertEqual(ee_deltas.shape, (2, 7))
            self.assertEqual(obj_poses.shape, (2, 7))
            self.assertTrue(
                len(timestamps)
                == len(proprio)
                == len(ee_poses)
                == len(ee_deltas)
                == len(obj_poses)
            )
            np.testing.assert_array_equal(ee_deltas[0], np.zeros(7))
            self.assertEqual(
                (output / "ee_columns.txt").read_text(encoding="utf-8"),
                "x,y,z,roll,pitch,yaw,gripper_state",
            )
            self.assertEqual(
                json.loads(
                    (output / "camera_info.json").read_text(encoding="utf-8")
                ),
                {
                    "fixed_camera_pos": collector.cam_pos,
                    "fixed_camera_target": collector.cam_target,
                    "fixed_camera_up": collector.cam_up,
                },
            )

    def test_failed_episode_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            collector = DataCollector(
                client=object(),
                camera_keys=[],
                save_dir=directory,
                hz=5.0,
                object_ids=[],
            )
            collector.success = False
            collector._timestamps = [0.2]
            collector.save()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()

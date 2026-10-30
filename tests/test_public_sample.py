"""Validate the small version-controlled demonstration excerpt."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "examples" / "t2_new_smooth_sample"


class PublicSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads((SAMPLE_DIR / "metadata.json").read_text())
        cls.archive = np.load(SAMPLE_DIR / "sample_episode.npz")

    @classmethod
    def tearDownClass(cls):
        cls.archive.close()

    def test_expected_schema(self):
        expected_shapes = {
            "timestamps": (4,),
            "proprio": (4, 75),
            "ee_poses": (4, 7),
            "ee_deltas": (3, 7),
            "obj_poses": (4, 14),
            "fixed_camera": (4, 240, 320, 3),
            "l_eye": (4, 240, 320, 3),
            "r_eye": (4, 240, 320, 3),
        }
        self.assertEqual(set(self.archive.files), set(expected_shapes))
        for key, shape in expected_shapes.items():
            self.assertEqual(self.archive[key].shape, shape, key)

    def test_observation_action_alignment(self):
        np.testing.assert_allclose(
            self.archive["ee_deltas"][:, :3],
            np.diff(self.archive["ee_poses"][:, :3], axis=0),
            rtol=0,
            atol=1e-7,
        )
        self.assertTrue(np.all(np.diff(self.archive["timestamps"]) > 0))
        self.assertTrue(
            np.all(np.isin(self.archive["ee_deltas"][:, 6], (0.0, 1.0)))
        )

    def test_metadata_indices_match_archive(self):
        source = self.metadata["source"]
        self.assertEqual(len(source["observation_indices"]), 4)
        self.assertEqual(len(source["action_indices"]), 3)
        self.assertEqual(self.metadata["task"], "t2_new_smooth")


if __name__ == "__main__":
    unittest.main()

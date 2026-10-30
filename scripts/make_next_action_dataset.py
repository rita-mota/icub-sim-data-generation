"""
Build next-action delta labels from already collected trajectories.

The original collector stores ee_deltas[t] as the transition from t-1 to t.
For next-action prediction, each sample at index t should instead use the
transition from t to t+1 as its label.

This script only changes ee_deltas.npy. It preserves the original file as
ee_deltas_wrong_time.npy in the same episode folder, then writes the corrected
next-action labels back to ee_deltas.npy. Observation files and videos are left
unchanged; the RLDS converter should use len(ee_deltas) as the episode length.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_next_ee_deltas(poses: np.ndarray) -> np.ndarray:
    """
    Compute action labels for pose[t] -> pose[t + 1].

    poses shape is (N, 7): x, y, z, roll, pitch, yaw, gripper.
    Returns shape (N - 1, 7).
    """
    if poses.ndim != 2 or poses.shape[1] < 7:
        raise ValueError(f"Expected ee_poses.npy with shape (N, 7+), got {poses.shape}")
    if len(poses) < 2:
        raise ValueError("Need at least two poses to compute next actions")

    next_deltas = np.zeros((len(poses) - 1, poses.shape[1]), dtype=poses.dtype)

    next_deltas[:, :3] = poses[1:, :3] - poses[:-1, :3]
    next_deltas[:, 6] = poses[1:, 6] - poses[:-1, 6]

    rotations = R.from_euler("xyz", poses[:, 3:6], degrees=False)

    # World-frame relative rotation: R_next = R_delta * R_curr
    r_curr = rotations[:-1]
    r_next = rotations[1:]
    r_delta = r_next * r_curr.inv()
    next_deltas[:, 3:6] = r_delta.as_euler("xyz", degrees=False).astype(poses.dtype)

    if poses.shape[1] > 7:
        next_deltas[:, 7:] = poses[1:, 7:] - poses[:-1, 7:]

    return next_deltas


def is_episode_dir(path: Path) -> bool:
    return (path / "ee_poses.npy").is_file()


def find_episode_dirs(input_dir: Path) -> list[Path]:
    if is_episode_dir(input_dir):
        return [input_dir]
    return sorted({pose_file.parent for pose_file in input_dir.rglob("ee_poses.npy")})


def write_conversion_info(episode_dir: Path, num_source_frames: int, num_actions: int) -> None:
    info = {
        "source_episode": str(episode_dir),
        "num_source_frames": int(num_source_frames),
        "num_actions": int(num_actions),
        "files_changed": ["ee_deltas.npy"],
        "backup_file": "ee_deltas_wrong_time.npy",
        "action_semantics": "ee_deltas[i] is the delta from ee_poses[i] to ee_poses[i + 1]",
    }
    with open(episode_dir / "next_action_conversion.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def convert_episode(episode_dir: Path, overwrite: bool = False) -> None:
    pose_file = episode_dir / "ee_poses.npy"
    delta_file = episode_dir / "ee_deltas.npy"
    backup_file = episode_dir / "ee_deltas_wrong_time.npy"

    poses = np.load(pose_file)
    next_deltas = compute_next_ee_deltas(poses)

    if backup_file.exists():
        if not overwrite:
            raise FileExistsError(f"{backup_file} already exists. Use --overwrite to replace it.")
        backup_file.unlink()

    if delta_file.exists():
        delta_file.rename(backup_file)
    else:
        print(f"WARNING: {delta_file} does not exist; writing corrected deltas without backup.")

    np.save(delta_file, next_deltas)
    write_conversion_info(episode_dir, num_source_frames=len(poses), num_actions=len(next_deltas))


def convert_dataset(input_dir: Path, overwrite: bool = False) -> None:
    episode_dirs = find_episode_dirs(input_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No ee_poses.npy files found under {input_dir}")

    for episode_dir in episode_dirs:
        convert_episode(episode_dir, overwrite=overwrite)
        print(f"Converted: {episode_dir}")

    print(f"\nDone. Converted {len(episode_dirs)} episode(s) under {input_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace ee_deltas.npy with next-action labels computed from ee_poses.npy."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Episode directory or dataset root containing episode folders with ee_poses.npy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing ee_deltas_wrong_time.npy backup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_dataset(args.input_dir, overwrite=args.overwrite)


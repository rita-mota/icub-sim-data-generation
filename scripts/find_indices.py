"""Locate arm-joint subsequences inside a recorded proprioception vector."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np



def find_subsequence(target, sequence, name):
    """Find the sequence window with the lowest mean absolute error."""
    window_size = len(target)
    best_idx = -1
    min_error = float("inf")

    for index in range(len(sequence) - window_size + 1):
        window = sequence[index:index + window_size]
        error = np.mean(np.abs(window - target))
        if error < min_error:
            min_error = error
            best_idx = index

    print(f"\n{name.upper()} ARM SEARCH RESULT")
    print(f"  Best index: {best_idx}")
    print(f"  Mean absolute error: {min_error:.6f}")
    print(f"  Slice: proprio[{best_idx}:{best_idx + window_size}]")
    if min_error >= 1e-3:
        print("  Warning: the dataset starting pose may differ from the simulator.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match simulator arm states inside a saved proprioception vector."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/t1.yaml"))
    parser.add_argument("--proprio", type=Path, required=True)
    args = parser.parse_args()

    from icub_pybullet.pycub import pyCub

    simulator = pyCub(config=str(args.config))
    recorded = np.load(args.proprio)

    find_subsequence(np.asarray(simulator.get_arm_joint_states("right")), recorded, "right")
    find_subsequence(np.asarray(simulator.get_arm_joint_states("left")), recorded, "left")


if __name__ == "__main__":
    main()

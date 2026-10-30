"""Extract the first proprioceptive observation from an RLDS dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save the first proprioceptive RLDS observation as a NumPy array."
    )
    parser.add_argument("dataset_dir", type=Path, help="TFDS builder directory.")
    parser.add_argument("--output", type=Path, default=Path("rlds_frame_0_proprio.npy"))
    parser.add_argument("--split", default="all", help="TFDS split to inspect.")
    args = parser.parse_args()

    import tensorflow_datasets as tfds

    dataset = tfds.builder_from_directory(str(args.dataset_dir)).as_dataset(split=args.split)
    first_episode = next(iter(dataset))
    first_step = next(iter(first_episode["steps"]))
    proprio = first_step["observation"]["proprio"].numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, proprio)
    print(f"Saved {proprio.shape} proprioception array to {args.output}")


if __name__ == "__main__":
    main()

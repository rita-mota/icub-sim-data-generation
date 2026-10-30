"""Check recorded episodes for array-length mismatches and timestamp gaps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CAMERAS = ("fixed_camera", "l_eye", "r_eye")


def camera_length(episode_dir: Path, camera: str) -> int:
    archive = episode_dir / f"{camera}_video.npz"
    if archive.exists():
        with np.load(archive) as data:
            return len(data[data.files[0]])
    image_dir = episode_dir / f"{camera}_images"
    if not image_dir.exists():
        return 0
    return sum(path.is_file() for path in image_dir.iterdir())


def scan_dataset_health(data_path: Path, tolerance_multiplier: float = 1.2) -> int:
    """Print a health report and return the number of suspicious episodes."""
    mismatched_lengths = []
    corrupted_timestamps = []
    total_episodes = 0
    for episode_dir in sorted(path for path in data_path.rglob("*") if path.is_dir()):
        actions_path = episode_dir / "ee_deltas.npy"
        timestamps_path = episode_dir / "timestamps.npy"
        if not actions_path.exists() or not timestamps_path.exists():
            continue
        total_episodes += 1
        try:
            lengths = {"actions": len(np.load(actions_path, mmap_mode="r"))}
            lengths.update({camera: camera_length(episode_dir, camera) for camera in CAMERAS})
            if len(set(lengths.values())) > 1:
                mismatched_lengths.append((episode_dir, lengths))
        except Exception as exc:
            print(f"Could not read arrays in {episode_dir}: {exc}")
        try:
            timestamps = np.load(timestamps_path)
            if len(timestamps) >= 2:
                deltas = np.diff(timestamps)
                expected = float(np.median(deltas))
                maximum = float(np.max(deltas))
                if expected > 0 and maximum > expected * tolerance_multiplier:
                    corrupted_timestamps.append((episode_dir, int(np.argmax(deltas)), expected, maximum))
        except Exception as exc:
            print(f"Could not read timestamps in {episode_dir}: {exc}")

    print(f"\nScanned {total_episodes} episodes")
    print(f"Length mismatches: {len(mismatched_lengths)}")
    for episode_dir, lengths in mismatched_lengths:
        print(f"  {episode_dir}: {lengths}")
    print(f"Timestamp gaps: {len(corrupted_timestamps)}")
    for episode_dir, index, expected, actual in corrupted_timestamps:
        print(f"  {episode_dir}: step {index}, expected about {expected:.4f}s, observed {actual:.4f}s")
    return len(mismatched_lengths) + len(corrupted_timestamps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--tolerance-multiplier", type=float, default=1.2)
    args = parser.parse_args()
    if not args.dataset_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.dataset_dir}")
    if args.tolerance_multiplier <= 1:
        parser.error("--tolerance-multiplier must be greater than 1")
    suspicious = scan_dataset_health(args.dataset_dir, args.tolerance_multiplier)
    raise SystemExit(1 if suspicious else 0)


if __name__ == "__main__":
    main()

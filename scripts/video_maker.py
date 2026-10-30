#!/usr/bin/env python3
"""
Convert a .npy or .npz array of video frames into a lightweight debug MP4.

Expected frame shapes:
  (T, H, W, 3)  RGB
  (T, H, W, 4)  RGBA, alpha dropped
  (T, H, W)     grayscale, converted to RGB

Usage:
  python np_video_preview.py /path/to/fixed_camera_video.npz
  python np_video_preview.py /path/to/video.npy --fps 10 --quality 3
  python np_video_preview.py /path/to/video.npz --key arr_0 -o debug.mp4
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def load_frames(path: Path, key: str | None) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if not data.files:
            raise ValueError(f"No arrays found in {path}")

        if key is None:
            key = data.files[0]

        if key not in data.files:
            raise ValueError(
                f"Key '{key}' not found in {path}. Available keys: {data.files}"
            )

        print(f"Loaded NPZ key: {key}")
        return data[key]

    if path.suffix == ".npy":
        return np.load(path)

    raise ValueError(f"Expected .npy or .npz file, got: {path.suffix}")


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)

    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, axis=-1)

    if frames.ndim != 4:
        raise ValueError(f"Expected frames shaped (T,H,W,C), got {frames.shape}")

    if frames.shape[-1] == 4:
        frames = frames[..., :3]

    if frames.shape[-1] != 3:
        raise ValueError(f"Expected 3 RGB channels in last dim, got {frames.shape}")

    if frames.dtype != np.uint8:
        frames = frames.astype(np.float32)
        if frames.max() <= 1.0:
            frames *= 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .npy/.npz video frames to a lightweight debug MP4."
    )
    parser.add_argument("input", type=Path, help="Path to .npy or .npz video frames")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--key", type=str, default=None, help="NPZ key to read")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=4, help="1-10, lower is smaller")
    args = parser.parse_args()

    frames = normalize_frames(load_frames(args.input, args.key))

    output = args.output
    if output is None:
        output = args.input.with_suffix(".debug.mp4")

    print(f"Frames: {frames.shape}, dtype={frames.dtype}")
    print(f"Writing: {output}")

    iio.imwrite(
        output,
        frames,
        fps=args.fps,
        codec="libx264",
        quality=args.quality,
        macro_block_size=8,
    )

    print("Done.")


if __name__ == "__main__":
    main()
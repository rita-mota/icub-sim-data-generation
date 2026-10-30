"""
Previous converter kept for reference below. It was useful as a first pass, but it
used non-standard feature names, did not explicitly finalize the LeRobot dataset,
and could trip TensorFlow CUDA initialization during CPU-only conversion.
"""

# --- Previous converter -----------------------------------------------------
# import tensorflow_datasets as tfds
# import numpy as np
# import shutil
# import tyro
# from pathlib import Path
#
# # LeRobot imports
# # Import the constant from its new utility file
# from lerobot.utils.constants import HF_LEROBOT_HOME
#
# # Keep the dataset builder import as is
# from lerobot.datasets.lerobot_dataset import LeRobotDataset
#
# def main(rlds_dir: str, repo_id: str, *, push_to_hub: bool = False):
#     """
#     Converts a single RLDS dataset to LeRobot format.
#     
#     Args:
#         rlds_dir: Path to the local RLDS dataset directory.
#         repo_id: The target Hugging Face repo name (e.g., 'username/t1_dataset').
#         push_to_hub: Whether to push the dataset to the HF Hub after conversion.
#     """
#     output_path = HF_LEROBOT_HOME / repo_id
#     
#     # Clean up existing local dataset if overwriting
#     if output_path.exists():
#         shutil.rmtree(output_path)
#         
#     print(f"Loading RLDS dataset from: {rlds_dir}")
#     builder = tfds.builder_from_directory(rlds_dir)
#     rlds_dataset = builder.as_dataset(split='train')
#     
#     # 1. PEEK: Dynamically determine the obj_pose shape from the first step
#     first_episode = next(iter(rlds_dataset))
#     first_step = next(iter(first_episode['steps']))
#     # Wrap it in tuple() to cast it to standard Python integers
#     dynamic_obj_pose_shape = tuple(first_step['observation']['obj_pose'].shape)
#     
#     print(f"Detected obj_pose shape: {dynamic_obj_pose_shape}")
#     
#     # 2. Initialize LeRobot schema with the dynamic shape
#     dataset = LeRobotDataset.create(
#         repo_id=repo_id,
#         fps=10, 
#         robot_type="custom_robot",
#         features={
#             "fixed_camera": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
#             "l_eye": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
#             "r_eye": {"dtype": "image", "shape": (3, 240, 320), "names": ["channels", "height", "width"]},
#             "proprio": {"dtype": "float32", "shape": (75,), "names": ["state"]},
#             "ee_pose": {"dtype": "float32", "shape": (7,), "names": ["state"]},
#             "obj_pose": {"dtype": "float32", "shape": dynamic_obj_pose_shape, "names": ["state"]},
#             "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
#             #"task": {"dtype": "string", "shape": (1,), "names": ["task"]},
#         },
#         image_writer_threads=10,
#         image_writer_processes=5,
#     )
#     
#     # 3. Iterate and convert
#     for episode_idx, episode in enumerate(rlds_dataset):
#         for step in episode['steps']:
#             language_instruction = step['language_instruction'].numpy().decode('utf-8')
#             
#             dataset.add_frame({
#                 "fixed_camera": step['observation']['fixed_camera'].numpy(),
#                 "l_eye": step['observation']['l_eye'].numpy(),
#                 "r_eye": step['observation']['r_eye'].numpy(),
#                 "proprio": step['observation']['proprio'].numpy(),
#                 "ee_pose": step['observation']['ee_pose'].numpy(),
#                 "obj_pose": step['observation']['obj_pose'].numpy(),
#                 "action": step['action'].numpy(),
#                 "task": step["language_instruction"].numpy().decode("utf-8"),
#             })
#             
#         dataset.save_episode()
#         print(f"Saved episode {episode_idx}")
#         
#     if push_to_hub:
#         dataset.push_to_hub(
#             tags=["pi0", "rlds_converted"],
#             private=True,
#             license="apache-2.0",
#         )
#
# if __name__ == "__main__":
#     tyro.cli(main)
# --- End previous converter -------------------------------------------------

import os
import shutil
from pathlib import Path
from typing import Any

# Dataset conversion is CPU work. This also avoids TensorFlow failing when the
# installed CUDA runtime is newer than the available NVIDIA driver.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


CAMERA_MAP = {
    "fixed_camera": "observation.images.fixed_camera",
    "l_eye": "observation.images.left_eye",
    "r_eye": "observation.images.right_eye",
}


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, tf.Tensor):
        return value.numpy()
    return np.asarray(value)


def as_float32(value: Any) -> np.ndarray:
    return as_numpy(value).astype(np.float32, copy=False)


def as_scalar_float32(value: Any) -> np.ndarray:
    return np.asarray([as_numpy(value).item()], dtype=np.float32)


def as_text(value: Any) -> str:
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_first_step(rlds_dataset: tf.data.Dataset) -> dict:
    for episode in rlds_dataset:
        for step in episode["steps"]:
            return step
    raise ValueError("The RLDS train split is empty.")


def image_feature_from_step(step: dict, source_key: str) -> dict:
    image = as_numpy(step["observation"][source_key])
    if image.ndim != 3:
        raise ValueError(f"Expected {source_key} image with 3 dims, got shape {image.shape}.")
    return {
        "dtype": "video",
        "shape": tuple(int(dim) for dim in image.shape),
        "names": ["height", "width", "channels"],
    }


def build_features(first_step: dict) -> dict:
    observation = first_step["observation"]
    features = {
        lerobot_key: image_feature_from_step(first_step, rlds_key)
        for rlds_key, lerobot_key in CAMERA_MAP.items()
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": tuple(as_float32(observation["proprio"]).shape),
                "names": ["state"],
            },
            "observation.ee_pose": {
                "dtype": "float32",
                "shape": tuple(as_float32(observation["ee_pose"]).shape),
                "names": ["state"],
            },
            "observation.object_pose": {
                "dtype": "float32",
                "shape": tuple(as_float32(observation["obj_pose"]).shape),
                "names": ["state"],
            },
            "observation.source_timestamp": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": tuple(as_float32(first_step["action"]).shape),
                "names": ["action"],
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
            "next.discount": {
                "dtype": "float32",
                "shape": (1,),
                "names": None,
            },
        }
    )
    return features


def generate_lerobot_frames(episode: dict):
    for step in episode["steps"]:
        observation = step["observation"]
        yield {
            "observation.images.fixed_camera": as_numpy(observation["fixed_camera"]),
            "observation.images.left_eye": as_numpy(observation["l_eye"]),
            "observation.images.right_eye": as_numpy(observation["r_eye"]),
            "observation.state": as_float32(observation["proprio"]),
            "observation.ee_pose": as_float32(observation["ee_pose"]),
            "observation.object_pose": as_float32(observation["obj_pose"]),
            "observation.source_timestamp": as_scalar_float32(observation["timestamp"]),
            "action": as_float32(step["action"]),
            "next.reward": as_scalar_float32(step["reward"]),
            "next.discount": as_scalar_float32(step["discount"]),
            "task": as_text(step["language_instruction"]),
        }


def validate_output(repo_id: str, root: Path) -> None:
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    if dataset.num_episodes == 0:
        raise ValueError(f"Converted dataset at {root} has 0 episodes.")
    sample = dataset[0]
    expected_keys = {
        "observation.images.fixed_camera",
        "observation.images.left_eye",
        "observation.images.right_eye",
        "observation.state",
        "action",
    }
    missing = expected_keys - set(sample)
    if missing:
        raise ValueError(f"Converted dataset is missing expected keys: {sorted(missing)}")
    print(f"Validated converted dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")


def main(
    rlds_dir: str,
    repo_id: str,
    *,
    fps: int = 10,
    root: Path | None = None,
    overwrite: bool = True,
    push_to_hub: bool = False,
    validate: bool = True,
):
    """
    Convert this thesis RLDS dataset to LeRobotDataset v3 format.

    Args:
        rlds_dir: Path to a prepared TFDS/RLDS dataset directory.
        repo_id: Hugging Face dataset repo id, e.g. "ritam/sim_t1".
        fps: Frame rate to store in LeRobot metadata.
        root: Optional local output directory. Defaults to HF_LEROBOT_HOME / repo_id.
        overwrite: Remove the output directory before conversion if it already exists.
        push_to_hub: Push the converted dataset to the Hugging Face Hub.
        validate: Load one sample after conversion to catch invalid parquet/video output.
    """
    output_path = root if root is not None else HF_LEROBOT_HOME / repo_id

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)

    print(f"Loading RLDS dataset from: {rlds_dir}")
    builder = tfds.builder_from_directory(rlds_dir)
    rlds_dataset = builder.as_dataset(split="train")
    first_step = get_first_step(rlds_dataset)
    features = build_features(first_step)

    print(f"Writing LeRobot dataset to: {output_path}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        robot_type="custom_robot",
        features=features,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    num_episodes = 0
    num_frames = 0
    try:
        for episode_idx, episode in enumerate(rlds_dataset):
            episode_frames = 0
            for frame in generate_lerobot_frames(episode):
                dataset.add_frame(frame)
                episode_frames += 1

            if episode_frames == 0:
                print(f"Skipping empty episode {episode_idx}")
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            num_episodes += 1
            num_frames += episode_frames
            print(f"Saved episode {episode_idx} ({episode_frames} frames)")
    finally:
        dataset.finalize()

    print(f"Finished conversion: {num_episodes} episodes, {num_frames} frames")

    if validate:
        validate_output(repo_id, output_path)

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lerobot", "rlds_converted", "thesis"],
            private=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

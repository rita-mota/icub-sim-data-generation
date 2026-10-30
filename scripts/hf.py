from pathlib import Path

from huggingface_hub import HfApi


DATASET_NAME = "sim_dataset_t5_smooth"
TFDS_NAME = "sim_dataset_t5_smooth"
VERSION = "1.0.0"
REPO_ID = f"ritamota/{DATASET_NAME}"
LOCAL_FOLDER = Path.home() / "tensorflow_datasets" / TFDS_NAME / VERSION
PATH_IN_REPO = f"{DATASET_NAME}/{VERSION}"

api = HfApi()

# Create the dataset repository on the hub (if it doesn't exist)
api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset", 
    private=True, # Set to False if you want it public
    exist_ok=True
)

# Upload the entire folder recursively
api.upload_folder(
    folder_path=str(LOCAL_FOLDER),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=PATH_IN_REPO,
    commit_message=f"Upload {DATASET_NAME} {VERSION}",
)
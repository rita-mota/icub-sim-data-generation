from pathlib import Path

from huggingface_hub import HfApi


DATASET_NAME = "icub_sim_dataset_t2_smooth"
TFDS_NAME = "sim_dataset_t2_smooth"
VERSION = "1.0.0"
REPO_ID = f"ritamota/{DATASET_NAME}"
LOCAL_FOLDER = Path.home() / "tensorflow_datasets" / TFDS_NAME / VERSION
PATH_IN_REPO = f"{DATASET_NAME}/{VERSION}"

api = HfApi()

print(f"Uploading chunked dataset from {LOCAL_FOLDER} to HF Hub...")

api.upload_folder(
    folder_path=str(LOCAL_FOLDER),
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=PATH_IN_REPO,
    # The magic parameter: this deletes any old files on the Hugging Face Hub 
    # that do not exist in your new local folder (wiping the old massive tfrecords)
    delete_patterns="*", 
    commit_message="Update dataset: Timestamps fixed, now delta reflects future ee pose"
)

print("Upload completely successful! Old files wiped, new chunks live.")
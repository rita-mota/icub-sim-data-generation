"""Upload a prepared dataset directory to the Hugging Face Hub.

The command is a dry run unless --yes is supplied. Authentication is delegated
to huggingface_hub; no token is stored in this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="For example: username/dataset")
    parser.add_argument("--path-in-repo", default=".")
    parser.add_argument("--commit-message", default="Upload dataset files")
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create the dataset repository if it does not exist.",
    )
    parser.add_argument(
        "--visibility",
        choices=("private", "public"),
        default="private",
        help="Visibility used only with --create.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete remote files absent from the local folder before upload.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute the remote operation. Without this flag, print the plan only.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Folder does not exist: {folder}")

    print("Hugging Face upload plan")
    print(f"  Local folder: {folder}")
    print(f"  Repository:   {args.repo_id}")
    print(f"  Remote path:  {args.path_in_repo}")
    print(f"  Create repo:  {args.create} ({args.visibility})")
    print(f"  Replace mode: {args.replace}")

    if not args.yes:
        print("Dry run only. Re-run with --yes to execute.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Install the optional dependency with: pip install huggingface-hub"
        ) from exc

    api = HfApi()
    if args.create:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.visibility == "private",
            exist_ok=True,
        )

    upload_options = {}
    if args.replace:
        upload_options["delete_patterns"] = "*"

    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        **upload_options,
    )
    print("Upload completed.")


if __name__ == "__main__":
    main()

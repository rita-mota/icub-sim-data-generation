import os
import json
import time
import argparse

def process_session(session_dir):
    """Iterates through all episodes in a session and updates the metadata."""
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a valid directory.")
        return

    # Find all episode subfolders (e.g., episode_001, episode_002)
    episode_folders = sorted([f for f in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, f))])
    
    print(f"Found {len(episode_folders)} folders in {session_dir}. Starting prompt generation...\n")

    for folder in episode_folders:
        episode_path = os.path.join(session_dir, folder)
        metadata_path = os.path.join(episode_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            continue
            
        # Read the current JSON
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        base_prompt = data.get("base_prompt")
        if not base_prompt:
            print(f"[{folder}] No base_prompt found. Skipping.")
            continue
            
        data["base_prompt"] = "Stack the red cube on top of the green cube."
        with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=4)
        data["all_variations"] = ["Stack the red cube on top of the green cube."]

        with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update metadata.json with LLM-generated prompts.")
    parser.add_argument(
        "--session", 
        type=str, 
        required=True, 
        help="Path to the master session directory (e.g., ./recordings/t1_session_xxx/)"
    )
    args = parser.parse_args()
    
    process_session(args.session)
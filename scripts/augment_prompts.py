import os
import json
import argparse
import random

def process_task(task_dir, prompts_file):
    """Iterates through all sessions and episodes to assign unique prompts."""
    if not os.path.isdir(task_dir):
        print(f"Error: {task_dir} is not a valid directory.")
        return

    # 1. Load and clean the prompts from the text file
    if not os.path.exists(prompts_file):
        print(f"Error: Prompts file '{prompts_file}' not found.")
        return
        
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = []
        for line in f:
            clean_line = line.strip()
            if clean_line:
                # Remove numbering (e.g., "1. Stack the red cube" -> "Stack the red cube")
                if clean_line[0].isdigit() and "." in clean_line[:5]:
                    clean_line = clean_line.split(".", 1)[1].strip()
                prompts.append(clean_line)

    # 2. Collect ALL episodes across ALL sessions
    all_episodes = []
    
    # Find all session folders inside the master task folder
    session_folders = sorted([f for f in os.listdir(task_dir) if os.path.isdir(os.path.join(task_dir, f))])
    
    for session in session_folders:
        session_path = os.path.join(task_dir, session)
        
        # Find all episode folders inside this specific session
        episodes = sorted([f for f in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, f))])
        
        for ep in episodes:
            all_episodes.append(os.path.join(session_path, ep))

    print(f"Found {len(session_folders)} sessions containing a total of {len(all_episodes)} episodes in {task_dir}.")
    print(f"Loaded {len(prompts)} unique prompts from {prompts_file}.")

    if not all_episodes:
        print("No episodes found to process. Exiting.")
        return

    # 3. Check for sufficient prompts
    if len(prompts) < len(all_episodes):
        print(f"WARNING: You have {len(all_episodes)} episodes but only {len(prompts)} prompts.")
        print("Some episodes will have to share prompts. Falling back to random choice with repetition.")
        allow_repetition = True
    else:
        allow_repetition = False
        # Shuffle the master list of prompts so the assignment is completely random
        random.shuffle(prompts)

    # 4. Assign prompts to every collected episode
    for index, episode_path in enumerate(all_episodes):
        metadata_path = os.path.join(episode_path, "metadata.json")
        
        if not os.path.exists(metadata_path):
            continue
            
        # Read the current JSON
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            
        # Select the prompt
        if allow_repetition:
            assigned_prompt = random.choice(prompts)
        else:
            assigned_prompt = prompts[index] # Guaranteed unique!
            
        # Update JSON
        data["episode_prompt"] = assigned_prompt
        
        # Keep a historical record of variations applied to this episode
        data.pop("all_variations", None)
            
        # Save JSON safely
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        # Print a clean log of what happened
        ep_name = os.path.basename(episode_path)
        sess_name = os.path.basename(os.path.dirname(episode_path))
        print(f"  └── [{sess_name} / {ep_name}] Assigned: {assigned_prompt}")

    print("\n All metadata files updated successfully across the entire task!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign pre-generated prompts to all episodes in a task.")
    parser.add_argument(
        "--task_dir", 
        "-t",
        type=str, 
        required=True, 
        help="Path to the master task directory (e.g., ./recordings/t2_smooth/)"
    )
    parser.add_argument(
        "--prompts", 
        "-p",
        type=str, 
        required=True, 
        help="Path to the .txt file containing the prompt variations."
    )
    args = parser.parse_args()
    
    process_task(args.task_dir, args.prompts)
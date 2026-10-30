#OI

import os
import json
import argparse
import numpy as np

# ==========================================
# YOUR NEW BASELINE CAMERA CONFIGURATION
# ==========================================
# Change these values to whatever you want your NEW center point to be!
NEW_BASE_POS    = [-0.7554, -0.0012, 0.5712] 
NEW_BASE_TARGET = [-0.2063, 0.0048, -0.2645]
NEW_BASE_UP     = [0.8356, -0.0232, 0.5488]
# ==========================================

def randomize_session_cameras(target_dir, pos_noise=0.05):
    """Crawls directories recursively to overwrite and jitter camera_info.json files."""
    if not os.path.isdir(target_dir):
        print(f"Error: {target_dir} is not a valid directory.")
        return

    updated_count = 0
    
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(target_dir):
        if "camera_info.json" in files:
            json_path = os.path.join(root, "camera_info.json")
            
            try:
                # 1. Read the file (just to keep its structure if it has other keys)
                with open(json_path, 'r') as f:
                    cam_info = json.load(f)
                
                # 2. Inject random uniform noise to the NEW X and Y positions
                # (Keeping Z locked to the new baseline Z, exactly like you had it!)
                noisy_new_pos = [
                    NEW_BASE_POS[0] + np.random.uniform(-pos_noise, pos_noise),
                    NEW_BASE_POS[1] + np.random.uniform(-pos_noise, pos_noise),
                    NEW_BASE_POS[2]
                ]
                
                # 3. Completely overwrite the old data with your new baseline + noise
                cam_info["fixed_camera_pos"]    = noisy_new_pos
                cam_info["fixed_camera_target"] = NEW_BASE_TARGET
                cam_info["fixed_camera_up"]     = NEW_BASE_UP
                
                # 4. Save it back down
                with open(json_path, 'w') as f:
                    json.dump(cam_info, f, indent=4)
                
                episode_name = os.path.basename(root)
                print(f"[{episode_name}] Overwritten & Shifted to: [{noisy_new_pos[0]:.3f}, {noisy_new_pos[1]:.3f}, {noisy_new_pos[2]:.3f}]")
                updated_count += 1
                
            except Exception as e:
                print(f"Error updating {json_path}: {e}")

    print(f"\n[Done] Successfully completely replaced and augmented camera positions for {updated_count} episodes!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overwrite and inject spatial diversity into recorded camera viewpoints.")
    parser.add_argument(
        "--path", "-p",
        required=True,
        type=str,
        help="Path to the master task folder or a single session folder containing episodes."
    )
    parser.add_argument(
        "--noise", "-n",
        type=float,
        default=0.05,
        help="Maximum camera displacement noise in meters (default: 0.05, which means +/- 5cm)"
    )
    args = parser.parse_args()
    
    randomize_session_cameras(args.path, args.noise)
import os
import argparse
import subprocess
import shutil
from pathlib import Path

def process_episode(ep_dir, action, force_re_render=False, draw_again=False):
    ep_path = Path(ep_dir)
    
    # 1. Validation: Is this actually a recorded episode?
    if not (ep_path / "timestamps.npy").exists():
        return # Skip, not a valid episode folder

    print(f"\n[{ep_path.name}] Processing...")
    
    # Check if .npz files already exist from a previous render
    npz_exists = len(list(ep_path.glob("*_video.npz"))) > 0

    # --- RENDER STEP ---
    if action in ['render', 'both']:
        if force_re_render:
            if os.path.exists(ep_path / "fixed_camera_images"):
                shutil.rmtree(ep_path / "fixed_camera_images", ignore_errors=False) 
            if os.path.exists(ep_path / "l_eye_images"):
                shutil.rmtree(ep_path / "l_eye_images", ignore_errors=False)
            if os.path.exists(ep_path / "r_eye_images"):
                shutil.rmtree(ep_path / "r_eye_images", ignore_errors=False)
        if os.path.exists(ep_path / "fixed_camera_images") and os.path.exists(ep_path / "r_eye_images") and os.path.exists(ep_path / "l_eye_images") and not force_re_render:
            print(f"[{ep_path.name}] Skipping Render: pngs already exist")
        else:
            print(f"[{ep_path.name}] Running pyCub Renderer...")
            # Call your existing render script
            result = subprocess.run(["python", "tools/render_recording.py", "--recording", str(ep_path)])
            if result.returncode != 0:
                print(f"[{ep_path.name}] ERROR during rendering. Skipping decoder.")
                return

    # --- DECODE STEP ---
    if action in ['decode', 'both']:
        # Refresh the check in case we just rendered them
        npz_exists = len(list(ep_path.glob("*_video.npz"))) > 0
        
        if not npz_exists:
            print(f"[{ep_path.name}] Skipping Decode: No .npz video files found. Run render first.")
        else:
            print(f"[{ep_path.name}] Running Decoder to extract images...")
            
            # Build the base command
            cmd = ["python", "tools/decoder.py", "--input_file", str(ep_path)]
            
            # Append the flag only if the user requested it
            if draw_again:
                cmd.append("--draw_again")
                
            # Run it EXACTLY once
            subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Batch process pyCub episodes (Render & Decode)")
    parser.add_argument("--dataset_dir", "-d", required=True, help="Path to main folder containing episodes")
    parser.add_argument("--action", "-a", choices=["render", "decode", "both"], default="both", 
                        help="What to do: render (npz), decode (images), or both.")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="Force re-rendering even if .npz files already exist.")
    parser.add_argument("--draw_again",action="store_true",
                        help = "Remove previous images in image folders and render them again")
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset_dir)

    if not dataset_path.exists():
        print(f"Error: Directory {dataset_path} does not exist.")
        return

    # Find all subdirectories
    episode_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    episode_folders.sort() # Keep them in order
    
    print(f"Found {len(episode_folders)} potential episode folders in {dataset_path.name}.")
    
    for ep_folder in episode_folders:
        process_episode(ep_folder, args.action, args.force, args.draw_again)

    print("\n[Done] All episodes processed.")

if __name__ == "__main__":
    main()
import os
import numpy as np
from PIL import Image
import argparse
import shutil
from pathlib import Path

def extract_npz_to_images(target_path: Path | str, draw_again) -> None:
    """
    Extracts .npz video files into image sequences.
    Safely handles both String and Pathlib inputs and searches recursively.
    """
    target_dir = Path(target_path)
    
    if not target_dir.exists():
        print(f"[Decoder] Directory not found: {target_dir}")
        return

    # 1. Find all .npz files recursively (fixes the depth quirk!)
    # This handles BOTH passing a root session folder OR a specific episode folder.
    npz_files = list(target_dir.rglob("*_video.npz"))
    
    if not npz_files:
        print(f"[Decoder] No video .npz files found in {target_dir.name}")
        return

    for npz_path in npz_files:
        # Extract camera name (e.g. 'fixed_camera' from 'fixed_camera_video.npz')
        camera_name = npz_path.name.replace("_video.npz", "")
        
        # Put the images in a folder right next to the .npz file
        output_folder = npz_path.parent / f"{camera_name}_images"
        
        # --- SKIP CHECK & FOLDER CLEARING ---
        if output_folder.exists() and any(output_folder.iterdir()):
            if not draw_again:
                print(f"[Decoder] Skipping {npz_path.name} (already extracted).")
                continue
            else:
                # Completely nuke the old folder so no ghost frames survive!
                print(f"[Decoder] Clearing old images from {output_folder.name}...")
                shutil.rmtree(output_folder)

        print(f"[Decoder] Extracting {npz_path.name} to {output_folder.name}...")
        output_folder.mkdir(parents=True, exist_ok=True)
        try:
            # Load the RAM-buffered block
            data = np.load(npz_path)
            if 'video' not in data:
                print(f"  [!] 'video' array not found in {npz_path.name}")
                continue
                
            video_frames = data['video'] # (N, H, W, 3) array

            # 2. Save each frame as a PNG using your existing PIL logic
            for i, frame in enumerate(video_frames):
                img = Image.fromarray(frame)
                filename = f"frame_{i:05d}.png"
                filepath = output_folder / filename
                img.save(filepath)
            
            print(f"  Successfully saved {len(video_frames)} frames.")
            
            # 1. We must explicitly close the numpy zip file so the OS lets us touch it
            data.close() 
            
            # 2. Delete the gigabytes of raw video data!
            # os.remove(npz_path)
            # print(f"  Deleted original {npz_path} to free up space.")

        except Exception as e:
            print(f"  Error processing {npz_path.name}: {e}")

# ---------------------------------------------------------
# The Argparse Guard
# Hides the CLI logic so it doesn't execute during imports
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PyBullet .npz videos to PNG frames.")
    parser.add_argument(
        "--input_file", 
        "-d",
        type=str, 
        default=".", 
        help="Path to the session or episode directory."
    )
    parser.add_argument("--draw_again", action="store_true",
                        help = "Remove previous images in image folders and render them again")
    args = parser.parse_args()
    
    extract_npz_to_images(args.input_file, args.draw_again)
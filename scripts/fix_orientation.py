import os
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def fix_orientation_deltas(base_dir: str):
    """
    Traverses the directory structure to find ee_poses.npy, 
    calculates correct World Frame deltas, and overwrites ee_deltas.npy.
    """
    base_path = Path(base_dir)
    
    # Recursively find all ee_poses.npy files in your nested architecture
    pose_files = list(base_path.rglob("ee_poses.npy"))
    
    if not pose_files:
        print(f"No 'ee_poses.npy' found in {base_dir}. Check your path!")
        return

    print(f"Found {len(pose_files)} episodes to process. Starting correction...\n")

    for pose_file in pose_files:
        episode_dir = pose_file.parent
        delta_file = episode_dir / "ee_deltas.npy"
        
        # 1. Load the absolute poses
        # Shape: (N, 7) -> [x, y, z, roll, pitch, yaw, gripper]
        poses = np.load(pose_file)
        
        if len(poses) < 2:
            print(f"Skipping {episode_dir.name} - not enough frames.")
            continue
            
        # 2. Backup the old buggy deltas just to be safe
        if delta_file.exists():
            backup_file = episode_dir / "ee_deltas_buggy_backup.npy"
            if not backup_file.exists(): # Don't overwrite the backup if we run this twice
                os.rename(delta_file, backup_file)

        # 3. Initialize the corrected delta array
        corrected_deltas = np.zeros_like(poses)
        
        # --- A. Translation & Gripper Deltas (Simple Subtraction) ---
        # positions: [x, y, z], gripper: [grip]
        corrected_deltas[1:, :3] = poses[1:, :3] - poses[:-1, :3]
        corrected_deltas[1:, 6] = poses[1:, 6] - poses[:-1, 6]
        
        # --- B. Orientation Deltas (Quaternion Math) ---
        # Extract Eulers (assumes roll, pitch, yaw = 'xyz' extrinsic)
        eulers = poses[:, 3:6]
        
        # Convert to Scipy Rotation objects (creates quaternions under the hood)
        rotations = R.from_euler('xyz', eulers, degrees=False)
        
        # R_curr = R_delta * R_prev  -->  R_delta = R_curr * R_prev^-1 (World Frame)
        r_prev = rotations[:-1]
        r_curr = rotations[1:]
        
        # Compute the precise relative rotation
        r_delta = r_curr * r_prev.inv()
        
        # Convert back to Euler for OpenVLA
        delta_eulers = r_delta.as_euler('xyz', degrees=False)
        
        # Insert into the delta array
        corrected_deltas[1:, 3:6] = delta_eulers
        
        # 4. Save the corrected deltas
        np.save(delta_file, corrected_deltas)
        print(f"Successfully corrected: {episode_dir.relative_to(base_path)}")

    print("\nAll episodes have been successfully corrected!")
    
if __name__ == "__main__":
    # Put your path here (try to use the absolute, full path just to be safe!)
    TARGET_DIRECTORY = "recordings/t1" 
    
    fix_orientation_deltas(TARGET_DIRECTORY)
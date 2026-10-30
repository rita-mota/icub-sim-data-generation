import os
import glob
import numpy as np

def scan_dataset_health(data_path, tolerance_multiplier=1.2):
    print(f"Scanning dataset at: {data_path}")
    print("Looking for end-of-episode mismatches and mid-episode drops...\n")
    
    mismatched_lengths = []
    corrupted_timestamps = []
    total_episodes = 0
    
    # os.walk automatically traverses every folder, no matter how deeply nested
    for root, dirs, files in os.walk(data_path):
        #print(f"\rCurrently checking: {root[-80:]: <80}", end="", flush=True)
        # Identify an episode folder by checking for core arrays
        if 'ee_deltas.npy' in files and 'timestamps.npy' in files:
            total_episodes += 1
            print(f"Episode number: {total_episodes} | Path: {root[-80:]}")
            
            # ==========================================
            # CHECK 1: Length Mismatches (End Drops)
            # ==========================================
            try:
                ee_len = len(np.load(os.path.join(root, 'ee_deltas.npy')))
                
                # Check fixed camera (handles both npz and raw images)
                if 'fixed_camera_video.npz' in files:
                    cam_npz = np.load(os.path.join(root, 'fixed_camera_video.npz'))
                    fixed_len = len(cam_npz[cam_npz.files[0]])
                else:
                    fixed_len = len(glob.glob(os.path.join(root, 'fixed_camera_images', '*.*')))
                    
                # Check left eye
                if 'l_eye_video.npz' in files:
                    cam_npz = np.load(os.path.join(root, 'l_eye_video.npz'))
                    l_eye_len = len(cam_npz[cam_npz.files[0]])
                else:
                    l_eye_len = len(glob.glob(os.path.join(root, 'l_eye_images', '*.*')))
                    
                # Check right eye
                if 'r_eye_video.npz' in files:
                    cam_npz = np.load(os.path.join(root, 'r_eye_video.npz'))
                    r_eye_len = len(cam_npz[cam_npz.files[0]])
                else:
                    r_eye_len = len(glob.glob(os.path.join(root, 'r_eye_images', '*.*')))

                lengths = {
                    'actions': ee_len,
                    'fixed_cam': fixed_len,
                    'l_eye': l_eye_len,
                    'r_eye': r_eye_len
                }

                # If turning the lengths into a set results in more than 1 unique number, they don't match
                if len(set(lengths.values())) > 1:
                    mismatched_lengths.append({'path': root, 'lengths': lengths})
                    
            except Exception as e:
                print(f"Error reading lengths in {root}: {e}")

            # ==========================================
            # CHECK 2: Timestamp Deltas (Mid Drops)
            # ==========================================
            try:
                timestamps = np.load(os.path.join(root, 'timestamps.npy'))
                if len(timestamps) >= 2:
                    deltas = np.diff(timestamps)
                    expected_delta = np.median(deltas)
                    max_delta = np.max(deltas)
                    
                    if max_delta > (expected_delta * tolerance_multiplier):
                        drop_index = np.argmax(deltas)
                        corrupted_timestamps.append({
                            'path': root,
                            'expected': expected_delta,
                            'actual_jump': max_delta,
                            'step_index': drop_index
                        })
            except Exception as e:
                print(f"Error reading timestamps in {root}: {e}")

    # ==========================================
    # PRINT DIAGNOSTIC REPORT
    # ==========================================
    print("\n" + "="*70)
    print(f"SCAN COMPLETE: Analyzed {total_episodes} total episodes")
    print("="*70)
    
    print(f"\n[TYPE 1] END-OF-EPISODE MISMATCHES: {len(mismatched_lengths)} episodes")
    print("Action required: None. Your TFDS builder script's min() function will safely truncate these.")
    for ep in mismatched_lengths:
        print(f"  - {ep['path']}")
        print(f"    Lengths found: {ep['lengths']}")
        
    print(f"\n[TYPE 2] MID-EPISODE DROPS (DESYNC): {len(corrupted_timestamps)} episodes")
    print("Action required: Delete these folders, or AI will learn from misaligned data.")
    for ep in corrupted_timestamps:
        print(f"  - {ep['path']}")
        print(f"    Drop at step {ep['step_index']}: Expected ~{ep['expected']:.4f}s gap, got {ep['actual_jump']:.4f}s")
    print("\n")

# Run the scanner
dataset_path = '/home/ritam/thesis/sim_data_collection/recordings/t2'
scan_dataset_health(dataset_path)
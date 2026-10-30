import numpy as np
from icub_pybullet.pycub import pyCub

# 1. Configure your paths
CONFIG_PATH = "/home/ritam/thesis/sim_data_collection/configs/t1.yaml"
DATASET_PATH = '/home/ritam/tensorflow_datasets/sim_dataset_t1/1.0.0'

def find_subsequence(target, sequence, name):
    """Slides a target array across a sequence to find the closest match."""
    window_size = len(target)
    best_idx = -1
    min_error = float('inf')
    
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i : i + window_size]
        # Calculate Mean Absolute Error between the window and our target
        error = np.mean(np.abs(window - target))
        
        if error < min_error:
            min_error = error
            best_idx = i
            
    # Output the results
    print(f"\n[*] {name.upper()} ARM SEARCH RESULT:")
    if min_error < 1e-3:
        print(f"    Match found at Index {best_idx} with near-zero error ({min_error:.6f})")
        print(f"    Your Python slice is: proprio[{best_idx}:{best_idx + window_size}]")
    else:
        print(f"    [!] Warning: Lowest error was {min_error:.6f}. The dataset starting pose")
        print(f"    might slightly differ from the simulator default pose.")
        print(f"    Best guess slice: proprio[{best_idx}:{best_idx + window_size}]")

def main():
    print("[*] Booting Simulator to grab target arm states...")
    sim = pyCub(config=CONFIG_PATH)
    sim_right = np.array(sim.get_arm_joint_states('right'))
    sim_left = np.array(sim.get_arm_joint_states('left'))
    
    print(f"[*] Loading first frame from RLDS dataset: {DATASET_PATH}")
    rlds_proprio = np.load("rlds_frame_0_proprio.npy")
    
    # Run the sliding window search
    find_subsequence(sim_right, rlds_proprio, "Right")
    find_subsequence(sim_left, rlds_proprio, "Left")

if __name__ == "__main__":
    main()
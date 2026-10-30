import numpy as np
import tensorflow_datasets as tfds

# Point this to your actual RLDS dataset directory
dataset_path = '/home/ritam/tensorflow_datasets/sim_dataset_t1/1.0.0'  # Update this path as needed
print("[*] Loading TFDS Dataset...")
dataset = tfds.builder_from_directory(dataset_path).as_dataset(split='all')
obs = next(iter(next(iter(dataset))['steps']))['observation']

# Save the 75D array to a raw numpy file
np.save("rlds_frame_0_proprio.npy", obs['proprio'].numpy())
print("[*] Successfully saved rlds_frame_0_proprio.npy!")
import os
import numpy as np

def remove_non_matching_arrays(root_dir, target_shape=(80, 80)):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                file_path = os.path.join(dirpath, filename)
                try:
                    array = np.load(file_path)
                    if array.shape != target_shape:
                        os.remove(file_path)
                        print(f"Removed {file_path} - shape {array.shape} does not match target {target_shape}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

# Example usage:
root_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays"
remove_non_matching_arrays(root_directory)

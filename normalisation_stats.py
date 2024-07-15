import numpy as np
import os
from glob import glob

def calculate_stats(directory_path):
    # Get all .npy files in the directory
    file_paths = glob(os.path.join(directory_path, '*.npy'))
    
    # Load and concatenate all arrays
    all_data = []
    for file_path in file_paths:
        arr = np.load(file_path)
        all_data.append(arr.flatten())  # Flatten each array
    
    combined_data = np.concatenate(all_data)
    
    # Calculate mean and standard deviation
    mean = np.mean(combined_data)
    std = np.std(combined_data)
    
    return mean, std

# Usage
directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/train'
mean, std = calculate_stats(directory_path)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
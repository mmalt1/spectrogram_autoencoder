import numpy as np
import os
from glob import glob

def calculate_stats(directory_path):
    """
    Calculates stats from all arrays in a directory (mean & std)    
    
    Args:
    directory_path (str): Path to the array directory
    
    Returns:
    mean (int): array directory mean
    std: array directory standard deviation
    """
    file_paths = glob(os.path.join(directory_path, '*.npy'))
    
    # load and concatenate all arrays
    all_data = []
    for file_path in file_paths:
        arr = np.load(file_path)
        all_data.append(arr.flatten())  

    combined_data = np.concatenate(all_data)
    
    mean = np.mean(combined_data)
    std = np.std(combined_data)
    
    return mean, std

directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/train'
mean, std = calculate_stats(directory_path)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
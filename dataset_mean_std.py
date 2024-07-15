import os
import numpy as np

import numpy as np
import os

def calculate_mean_std_from_npy(directory):
    sum_of_means = 0
    sum_of_squared_means = 0
    total_pixels = 0

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.endswith('.npy'):
            data = np.load(filepath)
            
            # Calculate mean and update sums
            mean_value = np.mean(data)
            sum_of_means += mean_value * data.size
            sum_of_squared_means += (mean_value ** 2) * data.size
            total_pixels += data.size

    # Calculate overall mean
    overall_mean = sum_of_means / total_pixels

    # Calculate overall standard deviation
    overall_var = (sum_of_squared_means / total_pixels) - (overall_mean ** 2)
    overall_std = np.sqrt(overall_var)

    return overall_mean, overall_std

# Example usage:
directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/val'
mean_result, std_result = calculate_mean_std_from_npy(directory_path)
print(f"Overall Mean: {mean_result}")
print(f"Overall Standard Deviation: {std_result}")

import os
import numpy as np

def calculate_mean_std_from_npy(directory):
    # Initialize lists to store mean and standard deviation
    means = []
    stds = []

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.endswith('.npy'):
            # Load data from .npy file
            data = np.load(filepath)

            # Calculate mean and standard deviation
            mean_value = np.mean(data)
            std_value = np.std(data)

            # Append to lists
            means.append(mean_value)
            stds.append(std_value)

    # Calculate overall mean and standard deviation if desired
    overall_mean = np.mean(means)/255.0
    overall_std = np.std(stds)/255.0 # dividing by 255.0 because assuming the image is an 8-bit image

    return overall_mean, overall_std

# Example usage:
directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/train'
mean_result, std_result = calculate_mean_std_from_npy(directory_path)
print(f"Overall Mean: {mean_result}")
print(f"Overall Standard Deviation: {std_result}")

import os
import numpy as np

def get_max_min_values_with_source(directory):
    all_values = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  # Assuming the arrays are stored as .npy files
            file_path = os.path.join(directory, filename)
            array = np.load(file_path)
            print(f'Loaded array: {filename}')
            # Store each value with its filename and index
            all_values.extend([(val, filename, idx) for idx, val in np.ndenumerate(array)])
    
    # Sort based on the value
    sorted_values = sorted(all_values, key=lambda x: x[0])
    
    # Get max 5 and min 5 values with their sources
    min_10 = sorted_values[:10]
    max_10 = sorted_values[-10:][::-1]  # Reverse to get descending order
    
    return min_10, max_10

# Usage
directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/test'
min_values, max_values = get_max_min_values_with_source(directory_path)

print("Minimum 5 values:")
for value, filename, index in min_values:
    print(f"Value: {value}, File: {filename}, Index: {index}")

print("\nMaximum 5 values:")
for value, filename, index in max_values:
    print(f"Value: {value}, File: {filename}, Index: {index}")
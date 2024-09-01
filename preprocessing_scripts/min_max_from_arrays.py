import os
import numpy as np

def get_max_min_values_with_source(directory):
    """
    Get the maximum and minimum 10 values of arrays in a directory. 

    Args:
    directory (str): directory of arrays

    Returns:
    min_10 (list): list of 10 minimum values in directory of arrays (int)
    max_10 (list): list of 10 maximum values in directory of arrays (int)
    """

    all_values = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.npy'): 
            file_path = os.path.join(directory, filename)
            array = np.load(file_path)
            print(f'Loaded array: {filename}')
            # store each value with its filename and index
            all_values.extend([(val, filename, idx) for idx, val in np.ndenumerate(array)])
    
    #sort based on the value
    sorted_values = sorted(all_values, key=lambda x: x[0])
    
    min_10 = sorted_values[:10]
    max_10 = sorted_values[-10:][::-1] 
    
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
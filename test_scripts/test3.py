import os
import numpy as np

def get_max_min_values(directory):
    all_values = []
    
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  # Assuming the arrays are stored as .npy files
            file_path = os.path.join(directory, filename)
            print(f'loaded array: {filename}')
            array = np.load(file_path)
            print('hello')
            print('here is the array: ', array)
            all_values.extend(array.flatten())
            print('array flattened')
        break
    # Convert to numpy array and sort
    all_values = np.array(all_values)
    sorted_values = np.sort(all_values)
    
    # Get max 5 and min 5 values
    min_5 = sorted_values[:10]
    max_5 = sorted_values[-10:][::-1]  # Reverse to get descending order
    
    return min_5, max_5

# Usage
directory_path = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/val/'
min_values, max_values = get_max_min_values(directory_path)

print("Minimum 5 values:", min_values)
print("Maximum 5 values:", max_values)

# array = np.load("/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/test/array_14_208_000004_000000.wav.npy")
# print(array)
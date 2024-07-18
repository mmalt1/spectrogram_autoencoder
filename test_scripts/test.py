import os
from pathlib import Path
import numpy as np

# def get_wav_files(root_dir, filetype):
#     wav_files = []
    
#     for dirpath, dirnames, filenames in os.walk(root_dir):
#         for filename in filenames:
#             if filename.endswith(filetype):
#                 file_path = os.path.join(dirpath, filename)
#                 wav_files.append(file_path)
    
#     return wav_files

def get_wav_files(root_dir, filetype):
    # The pattern "**" means all subdirectories recursively,
    # with "*.wav" meaning all files with any name ending in ".wav".
    file_list=[]  
    counter = 0 
    for file in Path(root_dir).glob(f"**/*{filetype}"):
        if not file.is_file():  # Skip directories
            continue

        with open(file, "r") as f:
            # print(f"{counter}: ", file)
            file_list.append(file)
        counter+=1
    return file_list



# Example usage:
root_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS/train-clean-360'
# other_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_array"
wav_files = get_wav_files(root_directory, '.wav')
# png_files = get_wav_files(other_directory, '.png')

# Print the list of WAV files
print("Number of wav files: ", len(wav_files))
# print("Number of png files: ", len(png_files))


# Define the directory containing the .npy files
# test_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/test'
# train_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/train'
# val_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val'

# npy_files = [f for f in os.listdir(test_directory) if f.endswith('.npy')]
# print("number of arrays in test data: ", len(npy_files))

# npy_files = [f for f in os.listdir(train_directory) if f.endswith('.npy')]
# print("number of arrays in train data: ", len(npy_files))


# npy_files = [f for f in os.listdir(val_directory) if f.endswith('.npy')]
# print("number of arrays in val data: ", len(npy_files))


# for f in os.listdir(input_directory):
#     full_path = os.path.join(input_directory, f)
#     g = np.load(full_path)
#     print("shape: ", g.shape)



# # # Ensure the output directory exists
# os.makedirs(output_directory, exist_ok=True)

# # Get a list of all .npy files in the directory
# npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]

# # Load the first array to compare with others
# if npy_files:
#     first_array = np.load(os.path.join(input_directory, npy_files[0]))
#     equal_files = [npy_files[0]]
#     print("SHAPE:", npy_files[0].shape)

#     # Compare all arrays with the first one
#     for npy_file in npy_files[1:]:
#         counter+= 1
#         current_array = np.load(os.path.join(input_directory, npy_file))
#         if np.array_equal(first_array, current_array):
#             equal_files.append(npy_file)
#         else:
#             print(f"Arrays are not equal: {counter}")

#     if len(equal_files) > 1:
#         print(f"Equal arrays: {', '.join(equal_files)}")
#         for file_name in equal_files:
#             # Copy the equal array files to the output directory
#             src_path = os.path.join(input_directory, file_name)
#             dest_path = os.path.join(output_directory, file_name)
#             np.save(dest_path, np.load(src_path))
#         print(f"All equal arrays saved to {output_directory}")
#     else:
#         print("No equal arrays found.")

# else:
#     print("No .npy files found in the directory.")

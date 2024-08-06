import os
import random
import shutil

def randomly_copy_files(source_dir, destination_dir, num_files):
    # Ensure destination directory exists
    counter = 0
    os.makedirs(destination_dir, exist_ok=True)

    # Get a list of all .pt files in the source directory
    pt_files = [file for file in os.listdir(source_dir) if file.endswith('.pt')]

    # Check if there are enough files to copy
    if len(pt_files) < num_files:
        raise ValueError(f"Not enough .pt files in the source directory. Found {len(pt_files)}, but need {num_files}.")

    # Randomly select the specified number of files
    selected_files = random.sample(pt_files, num_files)

    # Copy selected files to the destination directory
    for file in selected_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))
        counter += 1
        print(f"Copied file {counter}")

    print(f"Copied {num_files} files to {destination_dir}")

# Example usage
source_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev'
destination_directory = '/work/tc062/tc062/s2501147/autoencoder/evaluation_set'
number_of_files_to_copy = 499

randomly_copy_files(source_directory, destination_directory, number_of_files_to_copy)

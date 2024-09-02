"""
NOTE: used to create a directory of random 500 unseen tensors for testing the model. 
"""
import os
import random
import shutil

def randomly_copy_files(source_dir, destination_dir, num_files):
    
    counter = 0
    os.makedirs(destination_dir, exist_ok=True)
    pt_files = [file for file in os.listdir(source_dir) if file.endswith('.pt')]
    
    if len(pt_files) < num_files:
        raise ValueError(f"Not enough .pt files in the source directory. Found {len(pt_files)}, but need {num_files}.")

    selected_files = random.sample(pt_files, num_files)

    for file in selected_files:
        shutil.copy(os.path.join(source_dir, file), os.path.join(destination_dir, file))
        counter += 1
        print(f"Copied file {counter}")

    print(f"Copied {num_files} files to {destination_dir}")

source_directory = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev'
destination_directory = '/work/tc062/tc062/s2501147/autoencoder/evaluation_set'
number_of_files_to_copy = 499

randomly_copy_files(source_directory, destination_directory, number_of_files_to_copy)

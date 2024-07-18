import os
import json
import numpy as np

def load_processed_files(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            processed_files = json.load(f)
    else:
        processed_files = []
    return processed_files

def save_processed_files(log_file, processed_files):
    with open(log_file, 'w') as f:
        json.dump(processed_files, f)

def zero_arrays(input_directory, output_directory, log_file, nbr_columns=15, batch_size=10):
    """
    Zeroes out random columns of arrays to be used in in-painting training
    
    Parameters:
        input_directory (str): The directory containing the .npy files with spectrogram arrays
        output_directory (str): The directory where the zeroed columns .npy files will be saved
        log_file (str): The file to log processed files
        nbr_columns (int): The number of columns each array should have zeroed
        batch_size (int): The number of files to process before saving progress
    """
    processed_files = load_processed_files(log_file)
    counter = 0
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    print("Number of arrays: ", len(npy_files))

    for f in npy_files:
        zeroed_columns = np.random.choice(80, nbr_columns, replace=False)
        if f in processed_files:
            continue
        full_path = os.path.join(input_directory, f)
        array = np.load(full_path)
        
        array[:, zeroed_columns] = 0
        
        array_filename = f"{os.path.splitext(f)[0]}_zeroed.npy"
        array_path = os.path.join(output_directory, array_filename)
        np.save(array_path, array)
        print(f"Saved array {counter} with shape {array.shape} to {array_path}")

        processed_files.append(f)
        counter += 1
        print("COUNTER IN THIS BATCH: ", counter)
        # Save progress every `batch_size` files
        if counter % batch_size == 0:
            save_processed_files(log_file, processed_files)

    # Final save of progress
    save_processed_files(log_file, processed_files)

# Example usage:
input_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val"
output_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val_zeroed"
log_file = "/work/tc062/tc062/s2501147/autoencoder/processed_files_mask_log.json"

zero_arrays(input_directory, output_directory, log_file, nbr_columns=15, batch_size=10)

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

def chop_arrays(input_directory, output_directory, log_file, frames_per_chunk=80, batch_size=10):
    """
    Chops spectrogram arrays into smaller chunks with a specified number of frames.
    
    Args:
        input_directory (str): The directory containing the .npy files with spectrogram arrays.
        output_directory (str): The directory where the chopped .npy files will be saved.
        log_file (str): The file to log processed files.
        frames_per_chunk (int): The number of frames each chunk should have.
        batch_size (int): The number of files to process before saving progress.
    """
    processed_files = load_processed_files(log_file)
    counter = 0
    npy_files = [f for f in os.listdir(input_directory) if f.endswith('.npy')]
    print("Number of arrays: ", len(npy_files))
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    #     print("path exists")

    for f in npy_files:
        if f in processed_files:
            continue
        
        full_path = os.path.join(input_directory, f)
        spectrogram = np.load(full_path)
        
        num_frames = spectrogram.shape[1]
        # number of chunks to make with chunk size (80)
        num_chunks = num_frames // frames_per_chunk

        # make multiple chunks from each array
        
        for i in range(num_chunks):
            chunk = spectrogram[:, i * frames_per_chunk : (i + 1) * frames_per_chunk]
            chunk_filename = f"{os.path.splitext(f)[0]}_chunk_{i}.npy"
            chunk_path = os.path.join(output_directory, chunk_filename)
            np.save(chunk_path, chunk)
            print(f"Saved chunk {i} of {f} with shape {chunk.shape} to {chunk_path}")

         # if has more frames that don't fit in the chunks (with specified size)
        if num_frames % frames_per_chunk != 0:
            remaining_chunk = spectrogram[:, num_chunks * frames_per_chunk :]
            chunk_filename = f"{os.path.splitext(f)[0]}_chunk_{num_chunks}.npy"
            chunk_path = os.path.join(output_directory, chunk_filename)
            # np.save(chunk_path, remaining_chunk)
            print(f"Did not save remaining chunk of {f} with shape {remaining_chunk.shape} to {chunk_path}")
        
        processed_files.append(f)
        counter += 1
        print("COUNTER IN THIS BATCH: ", counter)
        # Save progress every `batch_size` files
        if counter % batch_size == 0:
            save_processed_files(log_file, processed_files)

    # Final save of progress
    save_processed_files(log_file, processed_files)

# Example usage:
input_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_train_arrays"
output_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_train_arrays"
log_file = "/work/tc062/tc062/s2501147/autoencoder/processed_train_files_chunk_log.json"

chop_arrays(input_directory, output_directory, log_file, frames_per_chunk=80, batch_size=10)

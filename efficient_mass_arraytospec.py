import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa.display


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

def make_spec_from_array(array_dir, new_spec_dir, log_file, sr, batch_size=10):
    processed_files = load_processed_files(log_file)
    counter = 0
    for array in os.scandir(array_dir):
        if array.is_file() and array.name.endswith('.npy') and array.name not in processed_files:
            spec_array = np.load(array.path)
            librosa.display.specshow(data=spec_array, sr=sr, cmap='gray_r')

            # Construct the output file name
            output_file_name = f"spec_{array.name}.png"
            output_file_path = os.path.join(new_spec_dir, output_file_name)
            plt.savefig(output_file_path)
            print(f"Saved spectrogram {output_file_name}")

            processed_files.append(array.name)
            counter += 1
            print("COUNTER FROM THIS BATCH: ", counter)
            # Save progress every `batch_size` files
            if counter % batch_size == 0:
                save_processed_files(log_file, processed_files)

    # Final save of progress
    save_processed_files(log_file, processed_files)

input_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays"
output_directory = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_spectrograms"
log_file = "/work/tc062/tc062/s2501147/autoencoder/processed_files_spec_log.json"

make_spec_from_array(array_dir=input_directory, new_spec_dir=output_directory, log_file=log_file, sr=24000, batch_size=10)
import os
import json
import shutil
from pathlib import Path

def load_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    return log

def save_log(log, log_path):
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)

filetype = "denoiser_aircon_20_output"
main_directory = "/work/tc062/tc062/s2501147/autoencoder/aircon1_wavs"
target_directory = f"/work/tc062/tc062/s2501147/autoencoder/aircon1_{filetype}_dns_dir"
noise = "aircon1"
log_path = f"/work/tc062/tc062/s2501147/autoencoder/{noise}_dns_dir_processed.json"
processed_files = load_log(log_path)

counter = 0

for root, dirs, files in os.walk(main_directory):
    print("Current directory:", root)
    
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for dir in dirs:
        print('Current speech sample: ', dir)
        subdirectory_name = dir
        
        if subdirectory_name in processed_files:
            print(f"Skipping {subdirectory_name}, already processed.")
            continue
        
        current_dir = f"{main_directory}/{subdirectory_name}/"
        
        for filename in os.listdir(current_dir):
            if filename.endswith(f'{filetype}.wav'):
                # filenm = Path(f'{filename}').stem
                target_wav_file_name = f"{subdirectory_name}_{filename}"
                wav_file = os.path.join(current_dir, filename)
                target_wav_file = os.path.join(target_directory, target_wav_file_name) 

                shutil.copy(wav_file, target_wav_file)
                
                counter += 1
                print(f"Counter: {counter}")
                print(f"Copied file {target_wav_file}")


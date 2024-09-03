import os
import librosa
import json
import shutil
import soundfile as sf

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

def downsample_files_in_dir(directory, save_array_dir, log_file, batch_size=10):
    processed_files = load_processed_files(log_file)
    counter = 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav') and filename not in processed_files:
                src_file = os.path.join(dirpath, filename)
                dst_file = os.path.join(save_array_dir, filename)
                # shutil.copy(src_file, dst_file)
                # Load the audio file
                y, sr = librosa.load(src_file, sr=None)  # Load at original 24000 Hz

                # Resample to 22050 Hz
                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)

                # Save the resampled audio
                sf.write(dst_file, y_resampled, 22050)

                processed_files.append(filename)
                counter += 1
                print(f"COUNTER FOR THIS BATCH: {counter}; copied file {filename}")
                # Save progress every `batch_size` files
                if counter % batch_size == 0:
                    save_processed_files(log_file, processed_files)

    # Final save of progress
    save_processed_files(log_file, processed_files)


current_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS_R/train-clean-360"
save_array_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_r/train/wavs"
log_file = "downsampled_ttsr_test.json"

downsample_files_in_dir(current_dir, save_array_dir, log_file)


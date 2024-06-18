
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

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


def wav_to_spec(directory, save_dir, save_array_dir):
    counter = 0
    # dir = os.fsencode(directory)
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav') and filename not in processed_files:
                file_path = os.path.join(dirpath, filename)
                
                # loading wav file with desired sampling rate
                wav, sr = librosa.load(file_path, sr=24000) # sr in LibriTTS = 24kHz
                # applying stft
                stft_wav = librosa.stft(wav,n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
                spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
                # extracting mel spectrogram with FastPitch regulations 
                mel_spec = librosa.feature.melspectrogram(S=spec, n_fft=2048, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
                librosa.display.specshow(mel_spec, sr=sr)
                
                np.save(f"{save_array_dir}/array_{filename}", mel_spec)
                plt.savefig(f"{save_dir}/mel_spectrogram{filename}.png")
                
                processed_files.append(filename)
                counter += 1

                # Save progress every `batch_size` files
                if counter % batch_size == 0:
                    save_processed_files(log_file, processed_files)

def make_spec_from_array(array_dir, new_spec_dir, sr):
    for array in os.scandir(array_dir):
        if array.is_file() and array.name.endswith('.npy'):
            spec_array = np.load(array.path)
        
        librosa.display.specshow(data=spec_array, sr=sr)

        # Construct the output file name
        output_file_name = f"spec_{array.name}.png"
        print("output_file_name: ", output_file_name)
        output_file_path = os.path.join(new_spec_dir, output_file_name)
        print("output_file_path: ", output_file_path)
        plt.savefig(f"{output_file_path}")


def chop_arrays(target_size_x, target_size_y, array_dir, chop_array_dir):
    
    for array in os.scandir(save_array_dir):
        if array.is_file() and array.name.endswith('.npy'):
            # Load the array from the file
            spec_array = np.load(array.path)
            rows, cols = spec_array.shape

            if rows < target_size_x or cols < target_size_y:
                raise ValueError("Array too small for desired chop")
            
            # centering the array with starting indices
            start_row = (rows - target_size_x)//2
            start_col = (cols - target_size_y)//2

            # centering array with end indices
            end_row = start_row + target_size_x
            end_col =  start_col + target_size_y

            # slicing array for center wav
            chopped_array = spec_array[start_row:end_row, start_col:end_col]

            # Construct the output file name
            output_file_name = f"80x80_{array.name}"
            print("output_file_name: ", output_file_name)
            output_file_path = os.path.join(chop_array_dir, output_file_name)
            print("output_file_path: ", output_file_path)

            # Save the chopped array
            np.save(outpSut_file_path, chopped_array)

def chunking_arrays(chunk_size, array_dir, chunk_dir):
    # and 80x80 array would give around 0.853 seconds
    for array in os.scandir(save_array_dir):
        if array.is_file() and array.name.endswith('.npy'):
            # Load the array from the file
            spec_array = np.load(array.path)
            rows, cols = spec_array.shape
            
            if cols != chunk_size or rows < chunk_size:
                raise ValueError("Array too small to chop")

            # number of chunks to make with chunk size (80)
            num_chunks = rows // chunk_size

            # make multiple chunks from each array
            for i in range(num_chunks):
                chunk = spec_array[i * chunk_size : (i+1)*chunk_size, :]
                chunk_filename = f"{os.path.splitext(f)[0]}_chunk_{i}.npy"
                chunk_path = os.path.join(output_directory, chunk_filename)
                np.save(chunk_path, chunk)
                print(f"Saved chunk {i} of {f} with shape {chunk.shape} to {chunk_path}")

            # if has more frames that don't fit in the chunks (with specified size)
            if num_frames % frames_per_chunk != 0:
                remaining_chunk = spectrogram[num_chunks * frames_per_chunk :, :]
                chunk_filename = f"{os.path.splitext(f)[0]}_chunk_{num_chunks}.npy"
                chunk_path = os.path.join(output_directory, chunk_filename)
                np.save(chunk_path, remaining_chunk)
                print(f"Saved remaining chunk of {f} with shape {remaining_chunk.shape} to {chunk_path}")



"""
MAKE SURE THAT ALL THE FILES HAVE DIFFERENT NAMES AAAHH
"""


current_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS/dev-clean"
save_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_spectrograms"
save_array_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_arrays"
save_chop = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays"
save_chop_spec = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_spectrograms"
log_file = "processed_files_log.json"

wav_to_spec(current_dir, save_dir, save_array_dir, log_file)
# chop_arrays(target_size_x=80, target_size_y=80, array_dir=save_array_dir, chop_array_dir=save_chop)
# make_spec_from_array(save_chop, save_chop_spec, 24000)

import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json

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

def wav_to_spec_array(directory, save_array_dir, log_file, batch_size=10):
    processed_files = load_processed_files(log_file)
    counter = 0
    
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.wav') and filename not in processed_files:
                file_path = os.path.join(dirpath, filename)
                
                # loading wav file with desired sampling rate
                wav, sr = librosa.load(file_path, sr=24000) # sr in LibriTTS = 24 kHz

                # applying stft
                stft_wav = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
                spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
                # extracting mel spectrogram with FastPitch regulations 
                mel_spec = librosa.feature.melspectrogram(S=spec, sr=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
                # np.ndarray [shape=(…, n_mels, t)]
                # librosa.display.specshow(mel_spec, sr=sr)
                
                np.save(f"{save_array_dir}/array_{filename}", mel_spec)
                # plt.savefig(f"{save_dir}/mel_spectrogram{filename}.png")
                
                processed_files.append(filename)
                counter += 1
                print("COUNTER FOR THIS BATCH: ", counter)
                # Save progress every `batch_size` files
                if counter % batch_size == 0:
                    save_processed_files(log_file, processed_files)

    # Final save of progress
    save_processed_files(log_file, processed_files)


current_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS_R/train-clean-360"
save_array_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/librittsR_fullspec"
log_file = "processed_tts_trainclean360.json"

wav_to_spec_array(current_dir, save_array_dir, log_file)


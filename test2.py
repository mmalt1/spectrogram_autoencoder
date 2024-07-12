import matplotlib.pyplot as plt
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from reconstructor import RAutoencoder
from autoencoder_spec import Autoencoder
import numpy as np
import os
import librosa
import librosa.display



# def visualize_tensor(array_path, nbr_columns):
#     og_tensor = np.load(array_path)
#     columns = random.sample(range(0,79), nbr_columns)
#     for column in columns:
#         og_tensor[:,:,:, column]=0
    
#     image_array = og_tensor[0,0,:,:]
#     a_max = image_array.min()
#     print("image array shape: ", image_array)
#     print("max of array: ", a_max)
#     plt.imshow(image_array,cmap='grey')
#     plt.savefig('test_zero_tensor.png')


def extract_mel_spectrogram(wav_file, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    # 1. Load the WAV file
    y, sr = librosa.load(wav_file, sr=sr)
    
    # 2. Resample (if needed) - this step is handled by librosa.load
    
    # 3. Convert to mono (if stereo) - also handled by librosa.load
    
    # 4. Perform STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Get magnitude spectrogram
    S = np.abs(D)
    
    # 5. Convert to mel scale
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_S = np.dot(mel_basis, S)
    
    # 6. Apply log transformation
    log_mel_S = librosa.power_to_db(mel_S)
    
    return log_mel_S

def extract_mel_spectrogram(wav_file, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, max_wav_value=32768.0):
    # Load and resample the audio
    wav, orig_sr = librosa.load(wav_file, sr=None)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)

    # Normalize
    wav = wav / max_wav_value

    # Compute STFT
    stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    magnitudes = np.abs(stft)

    # Mel spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spec = np.dot(mel_basis, magnitudes)

    # Log-scale mel spectrogram
    log_mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)

    return log_mel_spec 

# Usage
# test_wav = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav"
# tensor_dir = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels"

# log_mel_spectrogram = extract_mel_spectrogram(test_wav)
# log_mel_spectrogram_tensor = (torch.FloatTensor(log_mel_spectrogram))* 0.1
# torch.save(log_mel_spectrogram_tensor, f"{tensor_dir}/log_mel_spectrogram_try_01.pt")
# print('log mel spec tensor shape: ', log_mel_spectrogram_tensor.shape)



test_wav = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav"
tensor_dir = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels"

log_mel_spectrogram = extract_mel_spectrogram(test_wav)
log_mel_spectrogram_tensor = torch.tensor(log_mel_spectrogram)
torch.save(log_mel_spectrogram_tensor, f"{tensor_dir}/log_mel_spectrogram.pt")
print('log mel spec tensor shape: ', log_mel_spectrogram_tensor.shape)


max_value = 32768.0
print('Starting job')
wav, sr = librosa.load(test_wav, sr=None) # sr in LibriTTS = 24kHz
print('Starting sampling rate: ', sr)
print('Loaded wav file')
wav_resampled = librosa.resample(wav, orig_sr=sr, target_sr=22050)

wav_resampled = wav_resampled / max_value

# stft_wav = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
# stft_wav = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
stft_wav = librosa.stft(wav_resampled, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)

print('stft done')

spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
print('amplitude to db done')

# extracting mel spectrogram with FastPitch regulations 
mel_spec = librosa.feature.melspectrogram(S=spec, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
print('extracted mel spectrogram')

mel_spec_tensor = torch.tensor(mel_spec)
print('made into torch tensor')

torch.save(mel_spec_tensor, f"{tensor_dir}/test_resample_tensor.pt")
print("Audio tensor shape:", mel_spec_tensor.shape)


# test_tensor = "/work/tc062/tc062/s2501147/autoencoder/test_zeroed_tensors/zeroed_numpy_tensor_0.npy"
# visualize_tensor(test_tensor, 5)

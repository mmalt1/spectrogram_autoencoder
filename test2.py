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



def visualize_tensor(array_path, nbr_columns):
    og_tensor = np.load(array_path)
    columns = random.sample(range(0,79), nbr_columns)
    for column in columns:
        og_tensor[:,:,:, column]=0
    
    image_array = og_tensor[0,0,:,:]
    a_max = image_array.min()
    print("image array shape: ", image_array)
    print("max of array: ", a_max)
    plt.imshow(image_array,cmap='grey')
    plt.savefig('test_zero_tensor.png')



test_wav = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS/train-clean-100/19/198/19_198_000000_000002.wav"
tensor_dir = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels"

print('Starting job')
wav, sr = librosa.load(test_wav, sr=24000) # sr in LibriTTS = 24kHz
print('Starting sampling rate: ', sr)
print('Loaded wav file')

# stft_wav = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
stft_wav = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
print('stft done')

spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
print('amplitude to db done')

# extracting mel spectrogram with FastPitch regulations 
mel_spec = librosa.feature.melspectrogram(S=spec, sr=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
print('extracted mel spectrogram')

mel_spec_tensor = torch.tensor(mel_spec)
print('made into torch tensor')

torch.save(mel_spec_tensor, f"{tensor_dir}/test_2048_240000_tensor.pt")
print("Audio tensor shape:", mel_spec_tensor.shape)
print("Sample rate:", sr)

# test_tensor = "/work/tc062/tc062/s2501147/autoencoder/test_zeroed_tensors/zeroed_numpy_tensor_0.npy"
# visualize_tensor(test_tensor, 5)

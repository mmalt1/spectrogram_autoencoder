import torch
import torchaudio
import librosa
import numpy as np
import os, random

def wav_to_tensor(wav_file, tensor_dir, name):
    
    wav, sr = librosa.load(wav_file, sr=24000) # sr in LibriTTS = 24 kHz
    stft_wav = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
    spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
    mel_spec = librosa.feature.melspectrogram(S=spec, sr=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
    mel_spec= torch.tensor(mel_spec)
    mel_spec_tensor = torch.unsqueeze(mel_spec, dim=0)
    torch.save(mel_spec, f"{tensor_dir}/{name}.pt")
    
    return mel_spec_tensor


def add_noise_to_spec(spectrogram, noise_dir, tensor_dir, snr_db):
    """
    Add noise from a WAV file to a spectrogram tensor.
    
    Args:
    spectrogram (torch.Tensor): Input spectrogram of shape (channels, freq_bins, time_frames)
    noise_file_path (str): Path to the noise WAV file
    snr_db (float): Signal-to-Noise Ratio in decibels
    
    Returns:
    torch.Tensor: Noisy spectrogram
    """
    noise = random.choice(os.listdir(f"{noise_dir}"))
    noise_file = os.path.join(noise_dir, noise)
    noise_spectrogram = wav_to_tensor(noise_file, tensor_dir, f"noise_{noise}")
    

    # Ensure noise spectrogram has the same shape as input spectrogram
    if noise_spectrogram.shape != spectrogram.shape:
        noise_spectrogram = noise_spectrogram[:, :spectrogram.shape[1], :spectrogram.shape[2]]
        if noise_spectrogram.shape[2] < spectrogram.shape[2]:
            # If noise is shorter, repeat it
            repeats = spectrogram.shape[2] // noise_spectrogram.shape[2] + 1
            noise_spectrogram = noise_spectrogram.repeat(1, 1, repeats)[:, :, :spectrogram.shape[2]]
    
    # Calculate signal power
    signal_power = torch.mean(torch.abs(spectrogram)**2)
    
    # Calculate noise power
    noise_power = torch.mean(torch.abs(noise_spectrogram)**2)
    
    # Calculate scaling factor for noise
    snr = 10**(snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    
    # Scale and add noise to spectrogram
    noisy_spectrogram = spectrogram + scale * noise_spectrogram
    torch.save(noisy_spectrogram.squeeze(0), f"{tensor_dir}/noisy_audio.pt")
    return noisy_spectrogram

# when actually training, get random between 5 to 30 db to match Miipher
snr_db = 10

clean_audio = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/LibriTTS_R/train-clean-360/14/208/14_208_000001_000000.wav"
noisy_directory = "/work/tc062/tc062/s2501147/autoencoder/noise_train"
result_tensors = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels"

clean_spectrogram = wav_to_tensor(clean_audio, result_tensors, 'clean_audio')
noisy_spectrogram = add_noise_to_spec(clean_spectrogram, noisy_directory, result_tensors, snr_db)

print(f"Original spectrogram shape: {clean_spectrogram.shape}")
print(f"Noisy spectrogram shape: {noisy_spectrogram.shape}")
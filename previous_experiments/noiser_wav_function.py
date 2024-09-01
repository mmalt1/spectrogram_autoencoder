import torch
import torchaudio
import random
import os
import numpy as np

def add_noise_to_wav(clean_wav_path, noise_dir, snr_db, device='cuda'):
    """
    Add noise from a WAV file to another WAV file.
    
    Args:
    clean_wav_path (str): Path to the clean WAV file
    noise_dir (str): Directory containing noise WAV files
    snr_db (float): Signal-to-Noise Ratio in decibels
    device (str): Device to use for processing
    
    Returns:
    torch.Tensor: Noisy audio tensor
    int: Sample rate
    """
    clean_audio, sample_rate = torchaudio.load(clean_wav_path)
    clean_audio = clean_audio.to(device)

    noise_file = random.choice([f for f in os.listdir(noise_dir) if f.endswith('.wav')])
    print("Noise:", noise_file)
    noise_path = os.path.join(noise_dir, noise_file)
    noise_path = "/work/tc062/tc062/s2501147/autoencoder/noise_dataset/wavs/NeighborSpeaking_1.wav"
    
    noise_audio, noise_rate = torchaudio.load(noise_path)
    noise_audio = noise_audio.to(device)

    if noise_rate != sample_rate:
        noise_audio = torchaudio.functional.resample(noise_audio, noise_rate, sample_rate)

    if noise_audio.shape[1] >= clean_audio.shape[1]:
        start = random.randint(0, noise_audio.shape[1] - clean_audio.shape[1])
        noise_audio = noise_audio[:, start:start+clean_audio.shape[1]]
    else:
        # if noise is shorter, repeat it
        repeats = clean_audio.shape[1] // noise_audio.shape[1] + 1
        noise_audio = noise_audio.repeat(1, repeats)[:, :clean_audio.shape[1]]

    signal_power = torch.mean(clean_audio**2)
    noise_power = torch.mean(noise_audio**2)
    
    snr = 10**(snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    
    # scale and add noise to clean audio
    noisy_audio = clean_audio + scale * noise_audio

    return noisy_audio, sample_rate

clean_wav_path = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_dev_clean/wavs/84_121123_000007_000001.wav"
noise_dir = "/work/tc062/tc062/s2501147/autoencoder/noise_dataset/wavs"
save_dir = "/work/tc062/tc062/s2501147/autoencoder/test_noised_wavs"

snr_db = 5  # desired SNR in dB

noisy_audio, sample_rate = add_noise_to_wav(clean_wav_path, noise_dir, snr_db)
# If you want to save the noisy audio
torchaudio.save(f"{save_dir}/noisy_audio2.wav", noisy_audio.cpu(), sample_rate)
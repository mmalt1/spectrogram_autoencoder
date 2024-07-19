import torch
import torchaudio
import librosa
import numpy as np
import os, random


def wav_to_tensor(wav_file, name):
    
    wav, sr = librosa.load(wav_file, sr=24000) # sr in LibriTTS = 24 kHz
    stft_wav = librosa.stft(wav, n_fft=2048, hop_length=256, win_length=1024, window='hann', center=True, dtype=None, pad_mode='constant', out=None)
    spec = librosa.amplitude_to_db(abs(stft_wav), ref=np.max)
    mel_spec = librosa.feature.melspectrogram(S=spec, sr=24000, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0, power=1.0)
    mel_spec= torch.tensor(mel_spec)
    mel_spec_tensor = torch.unsqueeze(mel_spec, dim=0)
    # torch.save(mel_spec, f"{tensor_dir}/{name}.pt")
    
    return mel_spec_tensor


def add_noise_to_spec(spectrogram, noise_dir, snr_db, device='cuda'):
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
    # noise_spectrogram = wav_to_tensor(noise_file, f"noise_{noise}")
    noise_spectrogram = torch.load(noise_file)
    noise_spectrogram = noise_spectrogram.unsqueeze(0).unsqueeze(0)
    noise_spectrogram = noise_spectrogram.to(device)
    # noise_spectrogram = torch.load("/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/wav_files/mels/Typing_4.pt")
    # noise_spectrogram = noise_spectrogram.unsqueeze(0)
    # spectrogram = spectrogram.unsqueeze(0)
    
    # Ensure noise spectrogram has the same shape as input spectrogram
    if noise_spectrogram.shape != spectrogram.shape:

        if noise_spectrogram.shape[3] >= spectrogram.shape[3]:
            trd_dim_random_start = random.randint(0, (noise_spectrogram.shape[3] - spectrogram.shape[3]))
            trd_dim_random_end = trd_dim_random_start+spectrogram.shape[3]
            
            noise_spectrogram = noise_spectrogram[:, :, :,
                                                    trd_dim_random_start: trd_dim_random_end]

        # can eventually change to just "else:"
        if noise_spectrogram.shape[3] < spectrogram.shape[3]:
            # # If noise is shorter, repeat it
            repeats = spectrogram.shape[3] // noise_spectrogram.shape[3] + 1
            noise_spectrogram = noise_spectrogram.repeat(1, 1, 1, repeats)[:, :, :, :spectrogram.shape[3]]
    
    # Calculate signal power
    signal_power = torch.mean((spectrogram)**2)
    
    # Calculate noise power
    noise_power = torch.mean((noise_spectrogram)**2)
    
    # Calculate scaling factor for noise
    snr = 10**(snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    # print('spectrogram size: ', spectrogram.shape)
    # print('noisy spectrogram size: ', noise_spectrogram.shape)
    # print(f"Noise: {noise}")
    # Scale and add noise to spectrogram
    noisy_spectrogram = spectrogram + scale * noise_spectrogram

    return noisy_spectrogram

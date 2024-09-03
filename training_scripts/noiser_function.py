import torch
import os, random


def add_noise_to_spec(spectrogram, noise_dir, snr_db, device='cuda'):
    """
    Adds noise from a randomly selected noise file to the input spectrogram.

    The noise is adjusted to match the desired Signal-to-Noise Ratio (SNR) in decibels.
    The noise tensor is either truncated or repeated to match the dimensions of the input spectrogram.

    Args:
        spectrogram (torch.Tensor): Input log mel-spectrogram of shape (channels, freq_bins, time_frames)
        noise_file_path (str): Path to the noise log mel-spectrograms file (channels, freq_bins, time_frames)
        snr_db (float): Signal-to-Noise Ratio in decibels
        device (str): Device to perform computation ('cuda' by default)
    
    Returns:
        torch.Tensor: Noisy log mel-spectrogram with the same shape as the input
    """
    noise = random.choice(os.listdir(f"{noise_dir}"))
    noise_file = os.path.join(noise_dir, noise)
    noise_spectrogram = torch.load(noise_file)

    noise_spectrogram = noise_spectrogram.unsqueeze(0).unsqueeze(0)
    noise_spectrogram = noise_spectrogram.to(device)
    
    if noise_spectrogram.shape != spectrogram.shape:
        if noise_spectrogram.shape[3] > spectrogram.shape[3]:
            # if noise is longer, get random noise chunk of len(spectrogram)
            trd_dim_random_start = random.randint(0, (noise_spectrogram.shape[3] - spectrogram.shape[3]))
            trd_dim_end = trd_dim_random_start+spectrogram.shape[3]
            
            noise_spectrogram = noise_spectrogram[:, :, :,
                                                    trd_dim_random_start: trd_dim_end]

        # can eventually change to just "else:"
        if noise_spectrogram.shape[3] < spectrogram.shape[3]:
            # if noise is shorter, repeat it
            repeats = spectrogram.shape[3] // noise_spectrogram.shape[3] + 1
            noise_spectrogram = noise_spectrogram.repeat(1, 1, 1, repeats)[:, :, :, :spectrogram.shape[3]]
    
    # convert to linear scale
    spectrogram_linear = torch.exp(spectrogram)
    noise_spectrogram_linear = torch.exp(noise_spectrogram)

    signal_power = torch.mean(spectrogram**2)
    noise_power = torch.mean(noise_spectrogram**2)
    snr = 10**(snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))

    noisy_spectrogram_linear = spectrogram_linear + (scale * noise_spectrogram_linear)
    
    return torch.log(noisy_spectrogram_linear)

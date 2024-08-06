import torch
import os, random


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
    noise_spectrogram = torch.load(noise_file)

    noise_spectrogram = noise_spectrogram.unsqueeze(0).unsqueeze(0)
    noise_spectrogram = noise_spectrogram.to(device)
    
    # Ensure noise spectrogram has the same shape as input spectrogram
    if noise_spectrogram.shape != spectrogram.shape:

        if noise_spectrogram.shape[3] > spectrogram.shape[3]:
            # trd_dim_random_start = random.randint(0, (noise_spectrogram.shape[3] - spectrogram.shape[3]))
            trd_dim_start = 0
            trd_dim_end = trd_dim_start+spectrogram.shape[3]
            
            noise_spectrogram = noise_spectrogram[:, :, :,
                                                    trd_dim_start: trd_dim_end]

        # can eventually change to just "else:"
        if noise_spectrogram.shape[3] < spectrogram.shape[3]:
            # # If noise is shorter, repeat it
            repeats = spectrogram.shape[3] // noise_spectrogram.shape[3] + 1
            noise_spectrogram = noise_spectrogram.repeat(1, 1, 1, repeats)[:, :, :, :spectrogram.shape[3]]
    
    # Convert to linear scale
    spectrogram_linear = torch.exp(spectrogram)
    noise_spectrogram_linear = torch.exp(noise_spectrogram)

    # Calculate signal power
    signal_power = torch.mean(spectrogram**2)
    
    # Calculate noise power
    noise_power = torch.mean(noise_spectrogram**2)
    
    # Calculate scaling factor for noise
    snr = 10**(snr_db / 10)
    scale = torch.sqrt(signal_power / (snr * noise_power))
    # print('spectrogram size: ', spectrogram.shape)
    # print('noisy spectrogram size: ', noise_spectrogram.shape)
    # print(f"Noise: {noise}")
    # print(scale)
    noisy_spectrogram_linear = spectrogram_linear + (scale * noise_spectrogram_linear)
    
    return torch.log(noisy_spectrogram_linear)

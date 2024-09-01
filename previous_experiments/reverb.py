import torch
import torch.nn.functional as F

def generate_ir(length, decay_factor=0.01):
    """
    Generate an impulse response with exponential decay.
    
    Args:
    length (int): Length of the impulse response.
    decay_factor (float): Decay factor for the exponential decay.
    
    Returns:
    ir (torch.Tensor): Generated impulse response.
    """
    t = torch.arange(length)
    ir = torch.exp(-decay_factor * t)
    return ir

def apply_reverb_to_spectrogram(spectrogram, ir):
    """
    Apply reverb to a spectrogram by convolving each frequency band with the IR.
    
    Args:
    spectrogram (torch.Tensor): Input spectrogram (shape: [frequency, time]).
    ir (torch.Tensor): Impulse response (shape: [ir_length]).
    
    Returns:
    reverb_spectrogram (torch.Tensor): Spectrogram with reverb applied.
    """
    if ir.dim() != 1:
        raise ValueError("Impulse response must be a 1D tensor")
    
    num_freqs, num_times = spectrogram.shape
    print('Spectrogram shape: ', spectrogram.shape)
    reverb_spectrogram = torch.zeros(num_freqs, num_times + len(ir) - 1)
    print("reverb spec shape: ", reverb_spectrogram.shape)
    
    for i in range(num_freqs):
        reverb_spectrogram[i] = F.conv1d(spectrogram[i:i+1].unsqueeze(0), ir.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
    
    return reverb_spectrogram

print("JOB STARTING")
path = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000007_000001.pt"
save_dir = '/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels'
spectrogram = torch.load(path)
ir = generate_ir(spectrogram.shape[1], decay_factor=0.05)
reverb_spectrogram = apply_reverb_to_spectrogram(spectrogram, ir)
print('JOB FINISHED')
torch.save(reverb_spectrogram, f"{save_dir}/reverb_spec.pt")

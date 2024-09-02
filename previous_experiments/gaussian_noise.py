"""Adds Gaussian Noise to a speech spectrogram. Was used as an experiment for reducing vocoder artefacts
from model outputs. 
"""
import torch

original_tensor_path = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels/denoiser_0005_env_output.pt"
save_path = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels/denoiser_0005_env_output_addednoise.pt"

original_tensor = torch.load(original_tensor_path, map_location=torch.device('cpu'))
print('original tensor shape: ', original_tensor.shape)

mean = -2
std_deviation = 0.3


noise = torch.rand(original_tensor.shape) * std_deviation + mean
print("noise shape: ", noise.shape)

tensor_noised = original_tensor + noise
print("noised tensor shape: ", tensor_noised.shape)

torch.save(tensor_noised, save_path)
print('finished')
#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from reconstructor import RAutoencoder
from variable_length_restoration_skipconnections import VariableLengthRAutoencoder
import numpy as np
import os
from test_noiser_function import add_noise_to_spec

def load_and_preprocess_tensor(image_path, noise_directory, snr, save_dir):
    # spectrogram = np.load(image_path)
    spectrogram = torch.load(image_path, map_location=torch.device('cpu'))

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    # spectrogram = np.expand_dims(spectrogram, axis=0)
    # spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)

    # spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    print('size of spec tensor: ', spec_tensor.shape)

    noised_tensor = torch.clone(spec_tensor)
    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr, device='cpu')
    saving_tensor = noised_tensor.squeeze()
    torch.save(saving_tensor, f"{save_dir}/denoiser_speakers1_skip_input.pt")
    
    return noised_tensor

def predict_image_output(model, image_tensor, save_dir, input_path):
    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', image_tensor.shape)
        input = torch.load(input_path, map_location=torch.device('cpu'))
        output = model(image_tensor)
        flip_output = torch.flip(output, dims=[3])
        saved_output = flip_output.squeeze()
        print('saved ouput shape: ', saved_output.shape)
        saved_output = torch.flip(saved_output, dims=[1])
        print('saved ouput shape: ', saved_output.shape)
        torch.save(saved_output, f"{save_dir}/denoiser_speakers1_skip_output.pt")
        mse_loss = nn.MSELoss()(output, input)
        print("Mean Squared Error: ", mse_loss)
    
    return saved_output

def visualize_image(og_tensor, tensor_noised, predicted_image_tensor, save_dir):

    tensor_noised = tensor_noised.squeeze().numpy()
    
    predicted_image = predicted_image_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(3, 1)

    axs[0].imshow(og_tensor, cmap='gray')
    axs[0].set_title('Original Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].invert_yaxis()

    axs[1].imshow(tensor_noised, cmap='gray')
    axs[1].set_title('Noised Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].invert_yaxis()

    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Denoised Spectrogram')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Frequency')
    axs[2].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('denoiser_speakers1_skip.png')


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder(vae=False).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)

model.load_state_dict(torch.load("denoiser_speakers_skip.pt", map_location=torch.device('cpu')))

save_directory = '/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels'
noise_dir = "noise_dataset/mels/speakers_1_test"
# tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000007_000001.pt"
tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000008_000000.pt"
# tensor = "/work/tc062/tc062/s2501147/autoencoder/aircon1/5338_284437_000023_000001/denoiser_aircon_output.pt"
# og_tensor = torch.tensor(np.load(array))
og_tensor = torch.load(tensor, map_location=torch.device('cpu'))

noised_spec = load_and_preprocess_tensor(tensor, noise_dir, 15, save_directory)
predicted_image = predict_image_output(model, noised_spec, save_directory, tensor)
# # print("predicted digit: ", predicted_digit)

visualize_image(og_tensor, noised_spec, predicted_image, save_directory)
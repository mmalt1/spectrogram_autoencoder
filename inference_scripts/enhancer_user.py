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
from variable_length_restoration import VariableLengthRAutoencoder
import numpy as np
import os

def predict_image_output(model, save_dir, og_tensor):
    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', og_tensor.shape)
        torch.save(og_tensor, f"{save_dir}/enhancer__vae_custom_checkpoint4_input.pt")
        og_tensor = og_tensor.unsqueeze(0).unsqueeze(0)
        output, _, _ = model(og_tensor)
        flip_output = torch.flip(output, dims=[3])
        saved_output = flip_output.squeeze()
        print('saved ouput shape: ', saved_output.shape)
        saved_output = torch.flip(saved_output, dims=[1])
        print('saved ouput shape: ', saved_output.shape)
        torch.save(saved_output, f"{save_dir}/enhancer_vae_custom_checkpoint4_output.pt")
        mse_loss = nn.MSELoss()(output, og_tensor)
        print("Mean Squared Error: ", mse_loss)
    
    return saved_output

def visualize_image(enhanced_tensor, og_tensor, predicted_tensor, save_dir):
    
    torch.save(enhanced_tensor, f"{save_dir}/enhancer_vae_custom_checkpoint4_reference.pt")
    enhanced_image = enhanced_tensor.numpy()
    og_image = og_tensor.numpy()
    predicted_image = predicted_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(3, 1)

    axs[0].imshow(enhanced_image, cmap='gray')
    axs[0].set_title('LibriTTS-R Spectrogram')
    axs[0].axis('off')
    axs[0].invert_yaxis()

    axs[1].imshow(og_image, cmap='gray')
    axs[1].set_title('LibriTTS Spectrogram')
    axs[1].axis('off')
    axs[1].invert_yaxis()

    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Model Output Spectrogram')
    axs[2].axis('off')
    axs[2].invert_yaxis()
   
    plt.show()
    plt.savefig('enhancer_vae_custom_checkpoint4.png')


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder(vae=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)

model.load_state_dict(torch.load("enhancer_vae_custom_loss/checkpoint_4.pt"))

save_directory = '/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels'
tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000007_000001.pt"
enhanced_tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev_enhanced/84_121123_000007_000001.pt"
libritts_tensor = torch.load(tensor)
librittsr_tensor = torch.load(enhanced_tensor)

predicted_image = predict_image_output(model, save_directory, libritts_tensor)
# print("predicted digit: ", predicted_digit)

visualize_image(librittsr_tensor, libritts_tensor, predicted_image, save_directory)
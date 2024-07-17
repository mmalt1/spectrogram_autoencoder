#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

import matplotlib.pyplot as plt 
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from reconstructor import RAutoencoder
from variable_length_restoration import VariableLengthRAutoencoder
import numpy as np
import os
from noiser_function import add_noise_to_spec, wav_to_tensor

def load_and_preprocess_tensor(image_path, noise_directory, snr):
    spectrogram = np.load(image_path)
    # spectrogram = torch.load(image_path)

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    # spectrogram = spectrogram.unsqueeze(0)
    # spectrogram = spectrogram.unsqueeze(0)

    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    print('size of spec tensor: ', spec_tensor.shape)

    noised_tensor = torch.clone(spec_tensor)
    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr)

    return noised_tensor

def predict_image_output(model, image_tensor, save_dir):
    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', image_tensor.shape)
        output = model(image_tensor)
        flip1_output = torch.flip(output, dims=[2])
        # flip2_output = torch.flip(filp1_output, dims=[1])
        saved_output = flip1_output.squeeze()
        
    return saved_output

def visualize_image(og_tensor, tensor_noised, predicted_image_tensor, save_dir):

    tensor_noised = tensor_noised.squeeze().numpy()
    
    predicted_image = predicted_image_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(3, 1)

    axs[0].imshow(og_tensor, cmap='gray')
    axs[0].set_title('Original Spectrogram')
    axs[0].axis('off')
    axs[0].invert_yaxis()

    axs[1].imshow(tensor_noised, cmap='gray')
    axs[1].set_title('Noised Spectrogram')
    axs[1].axis('off')
    axs[1].invert_yaxis()
    
    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Denoised Spectrogram')
    axs[2].axis('off')

   
    plt.show()
    plt.savefig('denoiser_nonorm_cpt5_5db.png')


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)

model.load_state_dict(torch.load("denoiser_bigdata/checkpoint_5.pt"))

save_directory = 'torch_saved'
noise_dir = "/work/tc062/tc062/s2501147/autoencoder/noise_unseen"
array = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/test/array_14_208_000015_000002.wav.npy"
og_tensor = torch.tensor(np.load(array))

noised_spec = load_and_preprocess_tensor(array, noise_dir, 5)
predicted_image = predict_image_output(model, noised_spec, save_directory)
# print("predicted digit: ", predicted_digit)

visualize_image(og_tensor, noised_spec, predicted_image, save_directory)
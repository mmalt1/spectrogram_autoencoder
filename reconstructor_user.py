#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

import matplotlib.pyplot as plt
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from reconstructor import RAutoencoder
from autoencoder_spec import Autoencoder
import numpy as np
import os

def load_and_preprocess_tensor(image_path, save_dir):
    spectrogram = np.load(image_path)
    if spectrogram.shape != (80, 80):
        raise ValueError(f"Array at {image_path} has an incorrect shape: {spectrogram.shape}")

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    # dividing by 255.0 because assuming the image is an 8-bit image
    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    torch.save(spec_tensor.squeeze(), f"{save_dir}/saved23.pt")
    
    # make directly 1 column zeroed out for this batch 
    column = random.randint(0, 79)
    print("column: ", column)
    zeroed_tensor = torch.clone(spec_tensor)
    print("size of zeroed_tensor: ", zeroed_tensor.size())
    zeroed_tensor[:,:, :, column]=0


    return zeroed_tensor

def predict_image_output(model, image_tensor, save_dir):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        flip1_output = torch.flip(output, dims=[2])
        # flip2_output = torch.flip(filp1_output, dims=[1])
        saved_output = flip1_output.squeeze()
        torch.save(output.squeeze(), f"{save_dir}/saved22.pt")
    return saved_output

def visualize_image(tensor_masked, predicted_image_tensor, save_dir):

    # og_tensor = np.load(og_tensor_path)
    tensor_masked = tensor_masked.squeeze().numpy()
    # tensor_masked = tensor_masked.numpy()
    # # Plot the input and output images side by side
    # Convert the predicted tensor to a NumPy array
    # torch.save(predicted_image_tensor, f"{save_dir}/saved15.pt")
    print("predicted_image size: ", predicted_image_tensor.size())
    predicted_image = predicted_image_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].imshow(tensor_masked, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[0].invert_yaxis()
    
    axs[1].imshow(predicted_image, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')
    # axs[1].invert_yaxis()
    
    plt.show()
    plt.savefig('reconstructor_unseen_130000_flipagain.png')

print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = RAutoencoder().to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)
print("loaded autoenc")
model.load_state_dict(torch.load("reconstructor_130000.pt"))
print("loaded model")
save_directory = 'torch_saved'
img = load_and_preprocess_tensor("/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/test/array_84_121123_000011_000003.wav_chunk_1.npy", save_directory)
print(img.size())

predicted_image = predict_image_output(model, img, save_directory)
# print("predicted digit: ", predicted_digit)

visualize_image(img, predicted_image, save_directory)
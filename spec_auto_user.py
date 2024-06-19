#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

"""
NOTE
> ConvTranspose2d: can be seen as deconv layer
"""
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from autoencoder_spec import Autoencoder
import numpy as np
import os

def load_and_preprocess_tensor(image_path):
    spectrogram = np.load(image_path)
    if spectrogram.shape != (80, 80):
        raise ValueError(f"Array at {image_path} has an incorrect shape: {spectrogram.shape}")

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    # dividing by 255.0 because assuming the image is an 8-bit image
    spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    
    return spec_tensor

def predict_image_output(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    return output

def visualize_image(og_tensor_path, predicted_image_tensor, sr):

    og_tensor = np.load(og_tensor_path)
    # # Plot the input and output images side by side
    # Convert the predicted tensor to a NumPy array
    print("predicted_image size: ", predicted_image_tensor.size())
    predicted_image = predicted_image_tensor.squeeze().numpy()
    print("predicted_image size: ", predicted_image.shape)
    # flip the image for printing PLEASE
    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].imshow(og_tensor, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(predicted_image, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')
    
    plt.show()
    plt.savefig('spec_gpu_results7.png')

print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = Autoencoder().to(device)
print("loaded autoenc")
model.load_state_dict(torch.load("spec_autoencoder.pt"))
print("loaded model")
img = load_and_preprocess_tensor("/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val/array_84_121123_000008_000000.wav_chunk_2.npy")
print(img.size())

predicted_image = predict_image_output(model, img)
# print("predicted digit: ", predicted_digit)

visualize_image("/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val/array_84_121123_000008_000000.wav_chunk_2.npy", predicted_image, 24000)
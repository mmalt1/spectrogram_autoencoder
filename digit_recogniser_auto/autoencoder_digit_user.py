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
from autoencoder_cnn import Autoencoder
import numpy as np
import os

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with the same mean and std as MNIST
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict_image_output(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    return output

def visualize_image(image_path, predicted_image_tensor):
    image = Image.open(image_path).convert('L')

    # # Plot the input and output images side by side
     # Convert the predicted tensor to a NumPy array
    predicted_image = predicted_image_tensor.squeeze().cpu().numpy()
    
    # Unnormalize the predicted image
    predicted_image = (predicted_image * 0.3081) + 0.1307
    predicted_image = np.clip(predicted_image, 0, 1)  # Ensure the values are in [0, 1]

    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(predicted_image, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')
    
    plt.show()
    plt.savefig('gpu_results.png')

print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoenc_20epochs.pt"))
img = load_and_preprocess_image("handwritten_digits/mnist_7.png")
print(img.size())

predicted_image = predict_image_output(model, img)
# print("predicted digit: ", predicted_digit)

visualize_image("handwritten_digits/mnist_7.png", predicted_image)
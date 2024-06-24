import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from autoencoder_spec import Autoencoder
import numpy as np
import os


def load_zero_npy_arrays(npy_array, nbr_columns):
    array = np.load(npy_array)
    # change to for loop when going through directory
    zeroed_columns = np.random.choice(80, nbr_columns, replace=False)
    array[:, zeroed_columns] = 0

    return array

def visualize_image(og_array, zeroed_array):

    og_array = np.load(og_array)
    og_array = np.swapaxes(og_array, 0, 1)
    # # Plot the input and output images side by side
    # Convert the predicted tensor to a NumPy array
    zeroed_array = np.swapaxes(zeroed_array, 0, 1)


    fig, axs = plt.subplots(1, 2, figsize=(9, 5))
    axs[0].imshow(og_array, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    
    axs[1].imshow(zeroed_array, cmap='gray')
    axs[1].set_title('Zeroed Image')
    axs[1].axis('off')
    
    plt.show()
    plt.savefig('test_mask5.png')

# print("working directory: ", os.getcwd())
# os.chdir("/work/tc062/tc062/s2501147/autoencoder")
# print("working directory: ", os.getcwd())


input_array = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/train_zeroed/array_84_121123_000008_000000.wav_chunk_0_zeroed.npy"

zeroed_array = load_zero_npy_arrays(input_array, 15)

visualize_image(input_array, zeroed_array)




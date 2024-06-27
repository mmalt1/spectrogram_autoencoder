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


def visualize_tensor(array_path, nbr_columns):
    og_tensor = np.load(array_path)
    columns = random.sample(range(0,79), nbr_columns)
    for column in columns:
        og_tensor[:,:,:, column]=0
    
    image_array = og_tensor[0,0,:,:]
    a_max = image_array.min()
    print("image array shape: ", image_array)
    print("max of array: ", a_max)
    plt.imshow(image_array,cmap='grey')
    plt.savefig('test_zero_tensor.png')


test_tensor = "/work/tc062/tc062/s2501147/autoencoder/test_zeroed_tensors/zeroed_numpy_tensor_0.npy"
visualize_tensor(test_tensor, 5)

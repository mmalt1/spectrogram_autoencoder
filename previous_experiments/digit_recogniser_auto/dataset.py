#!/work/tc062/tc062/s2501147/myenv/bin/python
import torch
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Specify the data directory
data_dir = '/work/tc062/tc062/s2501147/autoencoder/preloaded_data'

# Load the training dataset
train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)

# Load the test dataset
test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

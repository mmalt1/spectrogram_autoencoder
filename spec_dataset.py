import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SpectrogramDataset(Dataset):
    def __init__(self, array_dir, transform=None):
        self.array_dir = array_dir
        self.file_list = [f for f in os.listdir(array_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.array_dir, self.file_list[idx])
        spectrogram = np.load(file_path)
        # if spectrogram.shape != (80, 80):
        #     raise ValueError(f"Array at {array_path} has an incorrect shape: {array.shape}")

        # Add channel dimension to make it (1, 80, 80) ie grayscale
        spectrogram = np.expand_dims(spectrogram, axis=0)
        # dividing by 255.0 because assuming the image is an 8-bit image
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram
        
def load_datasets(base_dir):
    transform = transforms.Compose([transforms.Normalize(0, 0.5)]) 
    # print("In load datasets")
    train_dataset = SpectrogramDataset(array_dir=os.path.join(base_dir, "train"), transform=transform)
    # print("train dataset loaded")
    val_dataset = SpectrogramDataset(array_dir=os.path.join(base_dir, "val"), transform=transform)
    # print("val dataset loaded")
    test_dataset = SpectrogramDataset(array_dir=os.path.join(base_dir, "test"), transform=transform)
    # print("test dataset loaded")
    
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # print("train loader done")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # print("val loader done")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # print("test loader done")
    return train_loader, val_loader, test_loader

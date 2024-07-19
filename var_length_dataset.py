import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class VarSpectrogramDataset(Dataset):
    def __init__(self, tensor_dir, transform=None):
        self.tensor_dir = tensor_dir
        self.file_list = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.tensor_dir, self.file_list[idx])
        spectrogram = torch.load(file_path).to(torch.float32)

        # Add channel dimension to make it (1, 80, time) ie grayscale
        spectrogram = torch.unsqueeze(spectrogram, 0)
        
        # get the length of the time dimension
        length = spectrogram.shape[2]

        if self.transform:
            spectrogram = self.transform(spectrogram)
            # print('spectrogram shapte after transform: ', spectrogram.shape)
        return spectrogram, length
        
def load_datasets(base_dir):
    transform = transforms.Compose([transforms.Normalize(0, 0.5)]) 
    # print("In load datasets")
    train_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "train"), transform=None)
    # print("train dataset loaded")
    dev_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "dev"), transform=None)
    # print("val dataset loaded")
    test_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "test"), transform=None)
    # print("test dataset loaded")
    
    return train_dataset, dev_dataset, test_dataset
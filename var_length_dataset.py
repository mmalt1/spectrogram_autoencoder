import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class VarSpectrogramDataset(Dataset):
    def __init__(self, array_dir, transform=None):
        self.array_dir = array_dir
        self.file_list = [f for f in os.listdir(array_dir) if f.endswith('.npy')]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.array_dir, self.file_list[idx])
        spectrogram = np.load(file_path)
        # spectrogram = torch.load(file_path)

        # Add channel dimension to make it (1, 80, time) ie grayscale
        spectrogram = np.expand_dims(spectrogram, axis=0)
        # spectrogram = torch.unsqueeze(spectrogram, 0)
        
        # get the length of the time dimension
        length = spectrogram.shape[2]
        # convert to tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        # print('spectrogram shape in dataloader: ', spectrogram.shape)

        if self.transform:
            spectrogram = self.transform(spectrogram)
            # print('spectrogram shapte after transform: ', spectrogram.shape)
        return spectrogram, length
        
def load_datasets(base_dir):
    transform = transforms.Compose([transforms.Normalize(0, 0.5)]) 
    # print("In load datasets")
    train_dataset = VarSpectrogramDataset(array_dir=os.path.join(base_dir, "train"), transform=None)
    # print("train dataset loaded")
    val_dataset = VarSpectrogramDataset(array_dir=os.path.join(base_dir, "val"), transform=None)
    # print("val dataset loaded")
    test_dataset = VarSpectrogramDataset(array_dir=os.path.join(base_dir, "test"), transform=None)
    # print("test dataset loaded")
    
    return train_dataset, val_dataset, test_dataset
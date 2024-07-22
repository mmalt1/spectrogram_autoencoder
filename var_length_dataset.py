import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from clipping_tensors_test import clip_to_equal_tensors

class VarSpectrogramDataset(Dataset):
    def __init__(self, tensor_dir, enhanced_tensor_dir, transform=None, enhancement=False):
        self.tensor_dir = tensor_dir
        self.enhanced_tensor_dir = enhanced_tensor_dir
        self.file_list = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        self.transform = transform
        self.enhancement = enhancement

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

        if self.enhancement:
            enhanced_file_path = os.path.join(self.enhanced_tensor_dir, self.file_list[idx])
            enhanced_spectrogram = torch.load(enhanced_file_path).to(torch.float32)
            enhanced_spectrogram = torch.unsqueeze(enhanced_spectrogram, 0)
            spectrogram, enhanced_spectrogram = clip_to_equal_tensors(spectrogram, enhanced_spectrogram)
            enhanced_length = enhanced_spectrogram.shape[2]
            # print("Enhanced shape: ", enhanced_spectrogram.shape)
            spectrogram_length = spectrogram.shape[2]
            # print("Spectrogram shape", spectrogram.shape)

            assert enhanced_length == spectrogram_length, f"Spectrogram & Enhanced Spectrogram are not the same length: {spectrogram.shape} & {enhanced_spectrogram.shape}"

            if self.transform:
                spectrogram = self.transform(spectrogram)
                enhanced_spectrogram = self.transform(enhanced_spectrogram)

            return spectrogram, enhanced_spectrogram, spectrogram_length


        return spectrogram, length
        
def load_datasets(base_dir):
    transform = transforms.Compose([transforms.Normalize(0, 0.5)]) 
    # print("In load datasets")
    train_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "train"),
                                          enhanced_tensor_dir=os.path.join(base_dir, "train_enhanced"),
                                          transform=None, enhancement=True)
    # print("train dataset loaded")
    dev_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "dev"),
                                        enhanced_tensor_dir=os.path.join(base_dir, "dev_enhanced"),
                                        transform=None, enhancement=True)
    # print("val dataset loaded")
    test_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "test"),
                                         enhanced_tensor_dir=os.path.join(base_dir, "test_enhanced"),
                                        transform=None, enhancement=True)
    # print("test dataset loaded")
    
    return train_dataset, dev_dataset, test_dataset
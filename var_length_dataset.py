import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from clipping_tensors_test import clip_to_equal_tensors

class VarSpectrogramDataset(Dataset):
    """
    A custom dataset class for loading spectrograms and their optional enhanced versions from disk.
    
    This dataset class handles loading spectrogram tensors from a specified directory, applying 
    optional transformations, and (if enhancement is enabled) loading corresponding enhanced 
    spectrograms.

    Attributes:
        tensor_dir (str): directory containing the original mid-quality spectrogram tensors
        enhanced_tensor_dir (str): directory containing the enhanced spectrogram tensors
        file_list (list): list of filenames of spectrogram tensors in directory
        transform (bool, optional): if True, applies a normalisation transform to the dataset
        enhancement (bool): if True, loads and returns both original and enhanced spectrograms with
                            lengths as a parallel dataset (tuple)

    Methods:
        __len__(): Returns the number of spectrogram files in tensor directory
        __getitem__(idx): returns the spectrogram(s) and their length at the specified index
    
    Note:
        The original mid-quality spectrograms and their corresponding enhanced spectograms have the
        same file name
    """

    def __init__(self, tensor_dir, enhanced_tensor_dir, transform=None, enhancement=False):
        """Initialises the dataset with directories and transformation options

        Args:
            tensor_dir (str): path to the directory with original mid-quality spectrograms
            enhanced_tensor_dir (str): path to the directory enhanced spectrograms
            transform (callable, optional): transformation function to apply to spectrograms. 
                                            Defaults to None.
            enhancement (bool, optional): flag to determine if enhanced spectrograms should be loaded.
                                          Defaults to False.
        """
        self.tensor_dir = tensor_dir
        self.enhanced_tensor_dir = enhanced_tensor_dir
        self.file_list = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
        self.transform = transform
        self.enhancement = enhancement

    def __len__(self):
        """Returns number of files in dataset

        Returns:
            int: number of spectrogram files
        """
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Loads and returns a spectrogram (and optionally its enhanced version) along with their lengths.

        This method loads a spectrogram tensor from the `tensor_dir`, optionally applies transformations, 
        and returns it. If `enhancement` is True, it also loads the corresponding enhanced spectrogram tensor 
        from `enhanced_tensor_dir`, ensures it has the same length as the original spectrogram, and returns both.

        Args:
            idx (int): index of spectrogram to retrieve

        Returns:
            tuple: If enhancement is False, tuple containing:
                    spectrogram (torch.Tensor): the spectrogram tensor with a channel dimension
                    length (int): length of the spectrogram
            if enhancement is True:
                tuple: tuple containing:
                        spectrogram (torch.Tensor): original mid-quality spectrogram tensor with a 
                                                    channel dimension
                        enhanced_spectrogram (torch.Tensor): ehanced spectrogram tensor with a channel
                                                             dimension
                        spectrogram_length (int): length of time dimension of the specotrogram and
                                                  enhanced_spectrogram
        Raises:
            AssertionError: if the spectrogram and enhanced_spectrogram are of different lengths after
                            clipping                                         
        """
        file_path = os.path.join(self.tensor_dir, self.file_list[idx])
        
        spectrogram = torch.load(file_path).to(torch.float32)

        # add channel dimension to make it (1, 80, time) ie grayscale
        spectrogram = torch.unsqueeze(spectrogram, 0)
        
        # get the length of the time dimension
        length = spectrogram.shape[2]

        if self.transform:
            spectrogram = self.transform(spectrogram)

        if self.enhancement:
            enhanced_file_path = os.path.join(self.enhanced_tensor_dir, self.file_list[idx])
            enhanced_spectrogram = torch.load(enhanced_file_path).to(torch.float32)
            enhanced_spectrogram = torch.unsqueeze(enhanced_spectrogram, 0)
            spectrogram, enhanced_spectrogram = clip_to_equal_tensors(spectrogram, enhanced_spectrogram)
            enhanced_length = enhanced_spectrogram.shape[2]
            spectrogram_length = spectrogram.shape[2]

            assert enhanced_length == spectrogram_length, f"Spectrogram & Enhanced Spectrogram are not the same length: {spectrogram.shape} & {enhanced_spectrogram.shape}"

            if self.transform:
                spectrogram = self.transform(spectrogram)
                enhanced_spectrogram = self.transform(enhanced_spectrogram)

            return spectrogram, enhanced_spectrogram, spectrogram_length


        return spectrogram, length
        
def load_datasets(base_dir):
    """Loads and returns the training, development and test datasets
    This function initialises and returns three instances of VarSpectrogrogramDataset for the training,
    development and test datasets, using the provided base directory. 

    Args:
        base_dir (str): path to the base directory containing the 'train', 'dev' and 'test' subdirectories
                        and the correspongin 'train_enhanced', 'dev_enhanced' and 'test_enhanced'
                        subdirectories to be used with the enhancement task

    Returns:
        tuple: tuple containing:
            train_dataset (VarSpectrogramDataset): Dataset for the training set, if enhancement is True,
                                                    parallel original and enhanced spectrogram dataset
            dev_dataset (VarSpectrogramDataset): Dataset for the development set, if enhancement is True,
                                                 parallel original and enhanced spectrogram dataset
            test_dataset (VarSpectrogramDataset): Dataset for the test set, if enhancement is True, 
                                                  parallel original and enhanced spectrogram dataset
    """
    transform = transforms.Compose([transforms.Normalize(0, 0.5)]) 
    train_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "train"),
                                          enhanced_tensor_dir=os.path.join(base_dir, "train_enhanced"),
                                          transform=None, enhancement=False)
    dev_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "dev"),
                                        enhanced_tensor_dir=os.path.join(base_dir, "dev_enhanced"),
                                        transform=None, enhancement=False)
    test_dataset = VarSpectrogramDataset(tensor_dir=os.path.join(base_dir, "test"),
                                         enhanced_tensor_dir=os.path.join(base_dir, "test_enhanced"),
                                        transform=None, enhancement=False)
    
    return train_dataset, dev_dataset, test_dataset
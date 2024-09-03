#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python
"""NOTE: used for inference for the inpainting task"""
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt 

from test_autoencoder import VariableLengthRAutoencoder

def load_and_preprocess_tensor(tensor_path, save_dir, nbr_columns):
    """Loads and preprocesses the tensors needed for inference. Zeroes out a specified number of columns
    (nbr_columns) in the tensor which will later be used as model input. 

    Args:
        tensor_path (str): path to .pt spectrogram tensor
        save_dir (str): path to save the masked tensor
        nbr_columns (int): number of columns to zeroe out

    Returns:
        torch.Tensor: zeroed_tensor, tensor with nbr_columns columns zeroed out
    """
    spectrogram = torch.load(tensor_path, map_location=torch.device('cpu'))
 
    # Add channel dimension to make it (1, 80, 80) ie grayscale
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)
    # torch.save(spec_tensor.squeeze(), f"{save_dir}/libriTTS_test_moreepochs_statsnorm_input.pt")

    columns = random.sample(range(0, 79), nbr_columns)
    print("columns: ", columns)
    zeroed_tensor = torch.clone(spec_tensor)
    print("size of zeroed_tensor: ", zeroed_tensor.size())
    for column in columns:
        zeroed_tensor[:,:, :, column]=0

    return zeroed_tensor

def predict_spectrogram_output(model, zeroed_spectrogram_tensor, save_dir):
    """Model inference, predicts an inpainted spectrogram from a masked spectrogram input using a
    trained VariableLengthRAutoencoder model.

    Args:
        model (VariableLengthRAutoencoder): trained model
        zeroed_spectrogram_tensor (torch.Tensor): zeroed spectrogram tensor
        save_dir (str): save path for tensor 

    Returns:
        _type_: _description_
    """
    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', zeroed_spectrogram_tensor.shape)
        output = model(zeroed_spectrogram_tensor)
        flip1_output = torch.flip(output, dims=[2])
        saved_output = flip1_output.squeeze()
        torch.save(saved_output, f"{save_dir}/quitepicnic_output.pt")
     
    return saved_output

def visualize_spectrograms(original, tensor_masked, predicted_spec_tensor, save_dir):
    """Plot for visualizing the input masked spectrogram, output inpainted spectrogram and the original
    non-masked spectrogram. 

    Args:
        original (torch.Tensor): original non-masked spectrogram
        tensor_masked (torch.Tensor): masked (zeroed out colums) spectrogram
        predicted_image_tensor (torch.Tensor): inpainted model output spectrogram
        save_dir (str): path to save plot 
    """

    og_tensor = torch.load(original, map_location=torch.device('cpu'))
    og_numpy = og_tensor.numpy()
    tensor_masked = tensor_masked.squeeze().numpy()
    predicted_image = predicted_spec_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(og_numpy, cmap='gray')
    axs[0].set_title('Original Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].invert_yaxis()

    axs[1].imshow(tensor_masked, cmap='gray')
    axs[1].set_title('Masked Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].invert_yaxis()
    
    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Inpainted Spectrogram')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Frequency')
    axs[2].invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig('masking_quitepicnic.png')


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder(debug=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)
print("loaded autoenc")
model.load_state_dict(torch.load("masking_quitepicnic.pt", map_location=torch.device('cpu')))
print("loaded model")
save_directory = 'torch_saved'

tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000007_000001.pt"


img = load_and_preprocess_tensor(tensor, save_directory, 5)

predicted_output_spectrogram = predict_spectrogram_output(model, img, save_directory)

visualize_spectrograms(tensor, img, predicted_output_spectrogram, save_directory)
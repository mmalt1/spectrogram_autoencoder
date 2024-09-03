#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

"""NOTE: Used for inference for the denoising task."""
import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from variable_length_restoration_skipconnections import VariableLengthRAutoencoder
from test_noiser_function import add_noise_to_spec

def load_and_preprocess_tensor(spectrogram, noise_directory, snr, save_dir):
    """Load and preprocesses the tensor necessary for inference. Noise is chosen randomly from a directory
    of noises and added to the speech spectrogram (spectrogram) with a designated signal to noise ratio
    (snr). 

    Args:
        spectrogram (torch.Tensor): spectrogram tensor to be loaded and preprocessed for inference
        noise_directory (str): directory of noise spectrogram tensors
        snr (int): Signal to Noise Ratio wanted when adding noise to speech 
        save_dir (str): path for saving the model input tensor
        noise (str): name of noise type, used for naming the save file

    Returns:
        torch.Tensor: noised_tensor, noisy speech spectrogram which will be used for inference
    """

    spectrogram = torch.load(spectrogram, map_location=torch.device('cpu'))

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)

    print('size of spec tensor: ', spec_tensor.shape)

    noised_tensor = torch.clone(spec_tensor)
    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr, device='cpu')
    saving_tensor = noised_tensor.squeeze()
    torch.save(saving_tensor, f"{save_dir}/denoiser_speakers1_skip_input.pt")
    
    return noised_tensor

def predict_spectrogram_output(model, noisy_speech_spectrogram, save_dir, input_path):
    """Inference, predicts the clean speech spectrogram output from the noisy speech spectrogram made in
    load_and_preprocess_tensor, using a VariableLengthRAutoencoder trained model. 

    Args:
        model (VariablLengthRAutoencoder): trained denoising model
        noisy_speech_spectrogram (torch.Tensor): noisy speech spectrogram to be used as model input
        save_dir (str): path to saving the denoised spectrogram

    Returns:
        torch.Tensor: saved_output, denoised speech spectrogram
    """

    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', noisy_speech_spectrogram.shape)
        input = torch.load(input_path, map_location=torch.device('cpu'))
        output = model(noisy_speech_spectrogram)
        flip_output = torch.flip(output, dims=[3])
        saved_output = flip_output.squeeze()
        print('saved ouput shape: ', saved_output.shape)
        saved_output = torch.flip(saved_output, dims=[1])
        print('saved ouput shape: ', saved_output.shape)
        torch.save(saved_output, f"{save_dir}/denoiser_speakers1_skip_output.pt")
        mse_loss = nn.MSELoss()(output, input)
        print("Mean Squared Error: ", mse_loss)
    
    return saved_output

def visualize_spectrograms(og_tensor, tensor_noised, predicted_denoised_tensor, save_dir):
    """Plot for visualizing the input noisy spectrogram, output denoised spectrogram and the original
    clean speech spectrogram. 

    Args:
        og_tensor (torch.Tensor): original non-masked spectrogram
        tensor_noised (torch.Tensor): noisy spectrogram
        predicted_denoised_tensor (torch.Tensor): denoised model output spectrogram
        save_dir (str): path to save plot 
    """

    tensor_noised = tensor_noised.squeeze().numpy()
    
    predicted_image = predicted_denoised_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(3, 1)

    axs[0].imshow(og_tensor, cmap='gray')
    axs[0].set_title('Original Spectrogram')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Frequency')
    axs[0].invert_yaxis()

    axs[1].imshow(tensor_noised, cmap='gray')
    axs[1].set_title('Noised Spectrogram')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Frequency')
    axs[1].invert_yaxis()

    axs[2].imshow(predicted_image, cmap='gray')
    axs[2].set_title('Denoised Spectrogram')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Frequency')
    axs[2].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    plt.savefig('denoiser_speakers1_skip.png')


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder(vae=False).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)

model.load_state_dict(torch.load("denoiser_speakers_skip.pt", map_location=torch.device('cpu')))

save_directory = '/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/torch_saved/mels'
noise_dir = "noise_dataset/mels/speakers_1_test"
tensor = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/dev/84_121123_000008_000000.pt"
og_tensor = torch.load(tensor, map_location=torch.device('cpu'))

noised_spec = load_and_preprocess_tensor(tensor, noise_dir, 15, save_directory)
denoised_spectrogram = predict_spectrogram_output(model, noised_spec, save_directory, tensor)

visualize_spectrograms(og_tensor, noised_spec, denoised_spectrogram, save_directory)
#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

import matplotlib.pyplot as plt 
import torch
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from reconstructor import RAutoencoder
from variable_length_restoration import VariableLengthRAutoencoder
import numpy as np
import os

def load_and_preprocess_tensor(image_path, save_dir, nbr_columns):
    # spectrogram = np.load(image_path)
    spectrogram = torch.load(image_path)
    # if spectrogram.shape != (80, 80):
    #     raise ValueError(f"Array at {image_path} has an incorrect shape: {spectrogram.shape}")

    # Add channel dimension to make it (1, 80, 80) ie grayscale
    # spectrogram = np.expand_dims(spectrogram, axis=0)
    # spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)

    # spec_tensor = torch.tensor(spectrogram, dtype=torch.float32)
    print('size of spec tensor: ', spec_tensor.shape)
    # torch.save(spec_tensor.squeeze(), f"{save_dir}/libriTTS_test_moreepochs_statsnorm_input.pt")

    columns = random.sample(range(0, 79), nbr_columns)
    print("columns: ", columns)
    zeroed_tensor = torch.clone(spec_tensor)
    print("size of zeroed_tensor: ", zeroed_tensor.size())
    for column in columns:
        zeroed_tensor[:,:, :, column]=0
    # torch.save(zeroed_tensor.squeeze(), f"{save_dir}/bigdata_zeroed_hifigan.pt")

    return zeroed_tensor

def predict_image_output(model, image_tensor, save_dir):
    model.eval()
    with torch.no_grad():
        print('size of image tensor: ', image_tensor.shape)
        output = model(image_tensor)
        flip1_output = torch.flip(output, dims=[2])
        # flip2_output = torch.flip(filp1_output, dims=[1])
        saved_output = flip1_output.squeeze()
        torch.save(output.squeeze(), f"{save_dir}/old_var_nonorm_checkpoin1_input.pt")
        print('size of output in predict image output: ', output.shape)
        print('size of flip1 output: ', flip1_output.shape)
        print('size of saved output: ', saved_output.shape)
    return saved_output

def visualize_image(tensor_masked, predicted_image_tensor, save_dir):

    # og_tensor = np.load(og_tensor_path)
    tensor_masked = tensor_masked.squeeze().numpy()
    # tensor_masked = tensor_masked.numpy()
    # # Plot the input and output images side by side
    # Convert the predicted tensor to a NumPy array
    torch.save(predicted_image_tensor, f"{save_dir}/old_var_nonorm_checkpoint1_output.pt")
    print("predicted_image size: ", predicted_image_tensor.size())
    predicted_image = predicted_image_tensor.numpy()
    print("predicted_image size: ", predicted_image.shape)

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(tensor_masked, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')
    axs[0].invert_yaxis()
    
    axs[1].imshow(predicted_image, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    axs[1].axis('off')
    # axs[1].invert_yaxis()
    
    plt.show()
    plt.savefig('old_var_length_restoration_nonorm_checkpoint1.png')


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
model.load_state_dict(torch.load("old_var_length_restoration_nonorm/checkpoint_1.pt"))
print("loaded model")
save_directory = 'torch_saved'
# npy_array
# wvg_pipeline_torch_tensor = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/wav_files/mels/14_208_000005_000000.pt"
# array = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS/test/array_14_208_000015_000002.wav.npy"
# torch_tensor = torch.from_numpy(np.load(array))
# print('size of torch tensor before load and preprocess: ', torch.load(torch_tensor).shape)

tensor = "/work/tc062/tc062/s2501147/FastPitch/FastPitches/PyTorch/SpeechSynthesis/FastPitch/wav_files/mels/14_208_000005_000000.pt"

img = load_and_preprocess_tensor(tensor, save_directory, 0)

predicted_image = predict_image_output(model, img, save_directory)
# print("predicted digit: ", predicted_digit)

visualize_image(img, predicted_image, save_directory)
#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python
"""NOTE: Mass inference in the denoising task, used for creating subjective listening test and objective
metric evaluation data. """

import os
import json
import torch
from variable_length_restoration import VariableLengthRAutoencoder
from test_noiser_function import add_noise_to_spec

def load_and_preprocess_tensor(spectrogram, noise_directory, snr, save_dir, noise):
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
    
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)

    print('size of spec tensor: ', spec_tensor.shape)
    noised_tensor = torch.clone(spec_tensor)
    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr, device='cpu')
    saving_tensor = noised_tensor.squeeze()
    torch.save(saving_tensor, f"{save_dir}/{noise}_noised_20_intput.pt")
    
    return noised_tensor

def predict_spectrogram_output(model, image_tensor, save_dir):
    """Inference, predicts the clean speech spectrogram output from the noisy speech spectrogram made in
    load_and_preprocess_tensor, using a VariableLengthRAutoencoder trained model. 

    Args:
        model (VariablLengthRAutoencoder): trained denoising model
        image_tensor (torch.Tensor): noisy speech spectrogram to be used as model input
        save_dir (str): path to saving the denoised spectrogram

    Returns:
        torch.Tensor: saved_output, denoised speech spectrogram
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        output = model(image_tensor)
        
        flip_output = torch.flip(output, dims=[3])
        saved_output = flip_output.squeeze()
        
        saved_output = torch.flip(saved_output, dims=[1])
        torch.save(saved_output, f"{save_dir}/{model_name}_20_output.pt")
    
    return saved_output

def load_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    return log

def save_log(log, log_path):
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=4)


print('STARTING JOB')
print("working directory: ", os.getcwd())
os.chdir("/work/tc062/tc062/s2501147/autoencoder")
print("working directory: ", os.getcwd())

device = torch.device("cpu")
print("device: ", device)


model = VariableLengthRAutoencoder(vae=False).to(device)
total_params = sum(p.numel() for p in model.parameters())
print("total params: ", total_params)
model_name = "denoiser_aircon"
common_path = "/work/tc062/tc062/s2501147/autoencoder"
noise = "aircon1"
speech = "/small_evaluation_set"
snr = 20

model.load_state_dict(torch.load(f"{model_name}.pt", map_location=torch.device('cpu')))

unseen_data_directory = f"{common_path}{speech}"
unseen_noise_directory = f"{common_path}/noise_dataset/mels/{noise}" 

log_path = f"processed_{noise}_{model_name}_20_log.json"
processed_files = load_log(log_path)
counter = 0

for filename in os.listdir(unseen_data_directory):
    base_name = os.path.splitext(filename)[0]
    save_directory = f"{common_path}/{noise}/{base_name}"
    
    if filename in processed_files:
        print(f"Skipping {filename}, already processed.")
        continue

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    file = os.path.join(unseen_data_directory, filename)
    
    if file.endswith('.pt'):
        og_tensor = torch.load(file, map_location=torch.device('cpu'))
        torch.save(og_tensor, f"{save_directory}/original.pt")
        noised_spec = load_and_preprocess_tensor(og_tensor, unseen_noise_directory, snr, save_directory, noise)
        predict_spectrogram_output(model, og_tensor, save_directory)
        
        processed_files.append(filename)
        save_log(processed_files, log_path)

    counter += 1
    print(f"Processed file {counter}")
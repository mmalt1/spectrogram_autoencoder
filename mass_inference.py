#!/work/tc062/tc062/s2501147/venv/mgpu_env/bin/python

import os
import json
import torch
from variable_length_restoration import VariableLengthRAutoencoder
from test_noiser_function import add_noise_to_spec

def load_and_preprocess_tensor(spectrogram, noise_directory, snr, save_dir, noise):
    
    spectrogram = spectrogram.unsqueeze(0)
    spec_tensor = spectrogram.unsqueeze(0)

    print('size of spec tensor: ', spec_tensor.shape)
    noised_tensor = torch.clone(spec_tensor)
    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr, device='cpu')
    saving_tensor = noised_tensor.squeeze()
    torch.save(saving_tensor, f"{save_dir}/{noise}_noised_20_intput.pt")
    
    return noised_tensor

def predict_image_output(model, image_tensor, save_dir):
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
model_name = "denoiser_aircon" # rename the model to not have checkpoint
common_path = "/work/tc062/tc062/s2501147/autoencoder"
noise = "aircon1"
# did speakers1, did aircon1, did env1, did station1, redo typing1
speech = "/small_evaluation_set"
snr = 20

model.load_state_dict(torch.load(f"{model_name}.pt", map_location=torch.device('cpu')))

unseen_data_directory = f"{common_path}{speech}"
unseen_noise_directory = f"{common_path}/noise_dataset/mels/{noise}" # GET NAME OF NOISE THAT WAS TAKEN, REMOVE RANDOM IN 

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
        predict_image_output(model, og_tensor, save_directory)
        
        processed_files.append(filename)
        save_log(processed_files, log_path)

    counter += 1
    print(f"Processed file {counter}")



"""
Output I want for Listening Test (not DNSMOS)
NOTE: print out counter & name of file everytime to keep track and make sure everyone has the same things
NOTE: shouldn't actually to visualise images, can do that in normal denoiser_user if needed

Environment model
    - input 
    - noised
        - noise dataset
    - denoised
        - noise dataset

Mix model
    - input
    - noised
        - noise dataset
    - denoised
        - noise dataset

One noise model
    - input
    - noised
        - noise dataset
    - denoised
        - noise dataset

Found data experiment
    - Model
        - input
        - noised
        - denoised

"""
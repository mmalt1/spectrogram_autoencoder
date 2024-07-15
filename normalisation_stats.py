import os
import numpy as np

def calculate_stats_from_spectrograms(directory):
    all_spectrograms = []

    for filename in os.listdir(directory):
        if filename.endswith('.npy'):  
            file_path = os.path.join(directory, filename)
            spectrogram = np.load(file_path)
            all_spectrograms.append(spectrogram)
    
    all_spectrograms = np.stack(all_spectrograms)
    
    mean = np.mean(all_spectrograms)
    std_dev = np.std(all_spectrograms)
    
    return mean, std_dev

spec_arrays_path = 'path/to/your/spectrogram/directory'
mean_value, std_dev_value = calculate_stats_from_spectrograms(spec_arrays_path)

print(f"Mean: {mean_value}")
print(f"Standard Deviation: {std_dev_value}")
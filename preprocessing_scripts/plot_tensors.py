import torch
import numpy as np
import matplotlib.pyplot as plt

spectrogram1_path = "/work/tc062/tc062/s2501147/autoencoder/torch_saved/bigdata3_unseen_input.pt"
spectrogram2_path = "/work/tc062/tc062/s2501147/autoencoder/torch_saved/bigdata3_unseen_output.pt"

spectrogram1 = torch.load(spectrogram1_path)
spectrogram2 = torch.load(spectrogram2_path)

flat_spec1 = spectrogram1.flatten()
flat_spec2 = spectrogram2.flatten()

counts1, bins1, _ = plt.hist(flat_spec1, bins=100, alpha=0.5, label='Input Spectrogram')
counts2, bins2, _ = plt.hist(flat_spec2, bins=100, alpha=0.5, label='Output Spectrogram')

sum_bars1 = np.sum(counts1)
sum_bars2 = np.sum(counts2)

plt.xlabel('Magnitude')
plt.ylabel('Frequency')
plt.title('Histogram Comparison of Two Spectrograms')
plt.legend()

# Add text annotations with the sum of bars
# plt.text(0.7, 0.95, f'Sum of Spectrogram 1 bars: {sum_bars1}', transform=plt.gca().transAxes)
# plt.text(0.7, 0.90, f'Sum of Spectrogram 2 bars: {sum_bars2}', transform=plt.gca().transAxes)

# Display the plot
plt.show()
plt.savefig('tensor_comparison_var.png')

print('Input spectrogram shape: ', spectrogram1.shape)
print('Output spectrogram shape: ', spectrogram2.shape)
print('Length of flattened input spectrogram: ', len(flat_spec1))
print('Length of flattene output spectrogram: ', len(flat_spec2) )
print('Sum of input spectrogram: ', sum_bars1)
print('Sum of output spectrogram: ', sum_bars2)
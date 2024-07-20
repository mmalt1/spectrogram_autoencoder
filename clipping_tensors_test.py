import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def align_tensors(tensor1, tensor2):
    if tensor1.shape[1] == tensor2.shape:
        print('The two tensors are the same shape')
        return tensor1, tensor2

    elif tensor1.shape[1] > tensor2.shape[1]:
        print('Tensor 1 is longer than Tensor 2')
        correlation = F.conv1d(
            tensor1.unsqueeze(0).float(),
            tensor2.flip(1).unsqueeze(0).float(),
            padding=tensor1.shape[1] - 1
        )

        offset = correlation.argmax() - (tensor1.shape[1] - 1)

        trimmed_tensor1 = tensor1[:, offset:offset + tensor2.shape[1]]

        return trimmed_tensor1, tensor2        

    elif tensor1.shape[1] < tensor2.shape[1]:
        print('Tensor 1 is shorter than Tensor 2')
        tensor1, tensor2 = tensor2, tensor1

        correlation = F.conv1d(
            tensor1.unsqueeze(0).float(),
            tensor2.flip(1).unsqueeze(0).float(),
            padding=tensor1.shape[1] - 1
        )

        offset = correlation.argmax() - (tensor1.shape[1] - 1)

        trimmed_tensor1 = tensor1[:, offset:offset + tensor2.shape[1]]

        return trimmed_tensor1, tensor2

# libri_tts = torch.randn(80, 90)
# libri_tts_r = torch.randn(80, 87)

# Align the tensors
# aligned_libri_tts, libri_tts_r = align_tensors(libri_tts, libri_tts_r)

# print("Aligned LibriTTS shape:", aligned_libri_tts.shape)
# print("LibriTTS-R shape:", libri_tts_r.shape)
    

libritts_path = "libritts_data/libriTTS_wg/dev/84_121123_000008_000002.pt"
librittsr_path = "libritts_data/libritts_r/dev/mels/84_121123_000008_000002.pt"

libritts_tensor = torch.load(libritts_path)
librittsr_tensor = torch.load(librittsr_path)

libritts_np = libritts_tensor.numpy()
librittsr_np = librittsr_tensor.numpy()

print('LibriTTS shape: ', libritts_tensor.shape)
print('LibriTTS-R shape: ', librittsr_tensor.shape)

fig, axs = plt.subplots(2, 1)

axs[0].imshow(libritts_np, cmap='gray')
axs[0].set_title('LibriTTS')
axs[0].axis('off')
axs[0].invert_yaxis()

axs[1].imshow(librittsr_np, cmap='gray')
axs[1].set_title('LibriTTS-R')
axs[1].axis('off')
axs[1].invert_yaxis()

plt.show()
plt.savefig('comparing_spectograms3.png')

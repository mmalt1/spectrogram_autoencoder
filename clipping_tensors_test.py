import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def clip_to_equal_tensors(libritts_tensor, librittsr_tensor):
    l_length = libritts_tensor.shape[2]
    lr_length =  librittsr_tensor.shape[2]
    
    if l_length == lr_length:
        return libritts_tensor, librittsr_tensor
    
    elif l_length < lr_length:
        difference = lr_length - l_length
        # print("difference: ", difference)
        if difference%2 == 0:
            clip = int(difference/2)
            # print('clip: ', clip)
            librittsr_tensor = librittsr_tensor[:, :, clip:lr_length-clip]
        else:
            difference = int(difference/2)
            beginning_clip = difference
            end_clip = difference + 1
            # print('beginning clip: ', beginning_clip)
            # print('end clip: ', end_clip)
            librittsr_tensor = librittsr_tensor[:, :, beginning_clip: lr_length - end_clip]
        
        return libritts_tensor, librittsr_tensor
    
    else:
        difference = l_length - lr_length
        # print("difference: ", difference)
        if difference%2 == 0:
            clip = int(difference/2)
            # print('clip: ', clip)
            libritts_tensor = libritts_tensor[:, :, clip:l_length-(clip)]
        else:
            difference = int(difference/2)
            beginning_clip = difference
            # print('beginning clip: ', beginning_clip)
            end_clip = difference + 1
            # print('end clip: ', end_clip)
            libritts_tensor = libritts_tensor[:, :, beginning_clip: l_length - end_clip]

        return libritts_tensor, librittsr_tensor


# libritts_path = "libritts_data/enhancement_dataset/dev/84_121123_000011_000003.pt"
# librittsr_path = "libritts_data/enhancement_dataset/dev_enhanced/84_121123_000011_000003.pt"

# libritts_t = torch.load(libritts_path)
# librittsr_t = torch.load(librittsr_path)

# libritts_t = torch.rand(1,80,545)
# librittsr_t = torch.rand(1,80,540)


# print('LibriTTS shape: ', libritts_t.shape)
# print('LibriTTS-R shape: ', librittsr_t.shape)

# libritts_tensor, librittsr_tensor = clip_to_equal_tensors(libritts_t, librittsr_t)

# libritts_np = libritts_tensor.numpy()
# librittsr_np = librittsr_tensor.numpy()

# print('LibriTTS shape: ', libritts_tensor.shape)
# print('LibriTTS-R shape: ', librittsr_tensor.shape)

# # fig, axs = plt.subplots(2, 1)

# axs[0].imshow(libritts_np, cmap='gray')
# axs[0].set_title('LibriTTS')
# axs[0].axis('off')
# axs[0].invert_yaxis()

# axs[1].imshow(librittsr_np, cmap='gray')
# axs[1].set_title('LibriTTS-R')
# axs[1].axis('off')
# axs[1].invert_yaxis()

# plt.show()
# plt.savefig('comparing_spectograms3_clipped.png')

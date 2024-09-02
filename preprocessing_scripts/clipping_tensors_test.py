import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def clip_to_equal_tensors(libritts_tensor, librittsr_tensor):
    """This function takes two spectrograms tensors and clips the longest one so they're of equal length.
    If the spectrograms are of equal length, directly returns them without modification. 
    If the spectrograms are not of equal length and the difference is even, an equal number of frames
    (corresponding to half of the difference) is subtracted from the start and end of the longer spectrogram
    If the difference is uneven, one more time frame is subtracted from the end than the beginning of the
    longer spectrogram. 

    Args:
        libritts_tensor (torch.Tensor): original mid-quality spectrogram tensor (channel, freq_bins, time)
        librittsr_tensor (torch.Tensor): restored high-quality spectrogram tensor (channel, freq_bins, time)

    Returns:
        tuple: tuple containing:
                libritts_tensor (torch.Tensor): original mid-quality spectrogram tensor, clipped if it was
                                                longer than the other spectrogram (same shape as input)
                librittsr_tensor (torch.Tensor): restored high-quality spectrogram tensor, clipped if it was
                                                 longer than the other spectrogram (same shape as input)
    """
    l_length = libritts_tensor.shape[2]
    lr_length =  librittsr_tensor.shape[2]
    
    if l_length == lr_length:
        return libritts_tensor, librittsr_tensor
    
    elif l_length < lr_length:
        difference = lr_length - l_length
        if difference%2 == 0:
            clip = int(difference/2)
            librittsr_tensor = librittsr_tensor[:, :, clip:lr_length-clip]
        else:
            difference = int(difference/2)
            beginning_clip = difference
            end_clip = difference + 1
            librittsr_tensor = librittsr_tensor[:, :, beginning_clip: lr_length - end_clip]
        
        return libritts_tensor, librittsr_tensor
    
    else:
        difference = l_length - lr_length
        if difference%2 == 0:
            clip = int(difference/2)
            libritts_tensor = libritts_tensor[:, :, clip:l_length-(clip)]
        else:
            difference = int(difference/2)
            beginning_clip = difference
            end_clip = difference + 1
            libritts_tensor = libritts_tensor[:, :, beginning_clip: l_length - end_clip]

        return libritts_tensor, librittsr_tensor
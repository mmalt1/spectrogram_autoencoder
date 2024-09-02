# Denoising Convolutional Autoencoder (DCAE)

This model uses a CNN Autoencoder to denoise speech spectrograms.
It can remove stationary and non-stationary noise, and can isolate main speakers from competing speakers. 

Text-to-speech (TTS) systems require substantial amounts of high-quality training data to produce natural-sounding synthetic voices, yet such data is expensive to record.
While open-source datasets offer an alternative, they often come with issues like background noise, reverberation, clipping, and limited bandwidth. Manually enhancing these datasets to studio quality can be equally costly.
Automatic speech restoration has emerged as a solution, but state-of-the-art models are complex and demand extensive training data, making them inaccessible to smaller teams relying on open-source data.

This Denoising Convolutional Autoencoder (DCAE, in variable_length_restoration.py) was created from scratch with PyTorch to explore smaller-scale denoising and restoration architectures for accessible TTS training data augmentation.

## Experiments
The DCAE architecture is first used for denoising speech spectrograms and speaker isolation. The best results are achieved using a U-Net inspired addition with skip connections.
This same architecture is then used with parallel mid-quality and enhanced training data for a fullt speech restoration task, but no conclusive results are found.
Full speech restoration is here defined as joint speech dereverberation, declipping, denoising and bandwidth extension.
The DCAE architecture can be extended with a Variational Autoencoder (VAE) component or a additional Transformer layers for the 

## Data
The LibriTTS and LibriTTS-R open-source speech datasets were used as training and testing data in this work. LibriTTS-R was used as parallel data to LibriTTS for the full speech restoration task. The open-source FastPitch preprocessing steps were used to transform the audio files into spectrograms, as they match many state-of-the-art vocoders. 

import argparse
import torch # type:ignore
import os
import math
import torch.nn as nn # type:ignore
import torch.nn.functional as F #type:ignore
import torch.optim as optim #type:ignore
import subprocess
import random
import numpy as np #type:ignore
from var_length_dataset import VarSpectrogramDataset, load_datasets
from torchvision import datasets, transforms #type:ignore
from torch.optim.lr_scheduler import StepLR #type:ignore
from torch.autograd import Variable #type:ignore
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  #type:ignore
from noiser_function import add_noise_to_spec, wav_to_tensor

comm_dir = "/work/tc062/tc062/s2501147/autoencoder/.wandb_osh_command_dir"

class VariableLengthRAutoencoder(nn.Module):
    """
    A Variable Length Restoration Autoencoder neural network class. 
    This class implements a spectrogram restoration/reconstruction CNN autoencoder, supporting variable
    length input, with an optional variational autoencoder (VAE) functionality. It supports residual
    (skip) connections and can be configured for debugging purposes.

    Args:
        deubg (bool, optional): If True, enables debugging for additional output. Defaults to False
        vae (bool, optional): If True, enable VAE functionality. Defaults to False. 

    Attributes:
        debug (bool): flag for debug mode
        vae (bool): flag for VAE mode
        enc1, enc2, enc3, enc4 (nn.Sequential): encoder layers
        dec1, dec2, dec3, dec4 (nn.Sequential): decoder layers
        mean_layer (nn.Linear): layer for computing mean layer from latent representation in VAE mode
        log_var_layer (nn.Linear): layer for computiong log variance from latent representation in VAE mode
        decoder_input (nn.Linear): layer for processing VAE latent space
        scaling (nn.Parameter): learnable scaling parameter 
        shifting (nn.Parameter): learnable shifting parameter 
    """
    def __init__(self, debug=False, vae=False):
        super(VariableLengthRAutoencoder, self).__init__()
        self.debug = debug
        self.vae = vae

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        # decoder that accepts skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512 + 256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64 + 1, 1, kernel_size=3, stride=1, padding=1))
        
        if self.vae:
            latent_dim = 512
            self.mean_layer = nn.Linear(latent_dim, 2)
            self.log_var_layer = nn.Linear(latent_dim, 2)
            self.decoder_input = nn.Linear(2, latent_dim)

        # custom scaling layer
        self.scaling = nn.Parameter(torch.FloatTensor([1.0]))
        self.shifting = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x):  
        """
        Forward pass of the autoencoder. 

        This method processes the input through the encoder and decoder, applying skip connections
        with a optional VAE processing. 

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, channels, height, width) where
                            height is the spectrogram's mel-frequency bins and width is time

        Returns:
            torch.Tensor or tuple: if VAE mode is disabled, returns the reconstructed input
                                   if VAE mode is enabled, returns a tuple 
                                   (reconstructed_input, mean, log_var) 
        """

        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        encoded = e4

        if self.vae:
            time_frames = encoded.shape[-1]
            encoded = encoded.mean(dim=-1)
            encoded = encoded.mean(dim=-1)
            encoded = torch.flatten(encoded, start_dim=1)
            mean, log_var = self.mean_layer(encoded), self.log_var_layer(encoded)
            var = log_var.exp()
            variance_tensor= torch.randn_like(var)
            sample = mean + var*variance_tensor
            resample = self.decoder_input(sample)
            resample = resample.unsqueeze(-1).unsqueeze(-1)
            resample_repeat = resample.repeat(1, 1, 20, time_frames) 
            downscaled = F.interpolate(x, (20, time_frames))
            decoder_input = torch.cat([resample_repeat, downscaled], 1)
            decoded = self.decoder(decoder_input)

        else:
            # decoder with skip connections
            d1 = self.dec1(torch.cat([encoded, F.interpolate(e3, size=encoded.shape[2:], mode='bilinear', align_corners=False)], dim=1))
            d2 = self.dec2(torch.cat([d1, F.interpolate(e2, size=d1.shape[2:], mode='bilinear', align_corners=False)], dim=1))
            d3 = self.dec3(torch.cat([d2, F.interpolate(e1, size=d2.shape[2:], mode='bilinear', align_corners=False)], dim=1))
            decoded = self.dec4(torch.cat([d3, F.interpolate(x, size=d3.shape[2:], mode='bilinear', align_corners=False)], dim=1))

        scaled = decoded * self.scaling + self.shifting 
    
        resized = F.interpolate(scaled, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        # print(f"resized allocated: { torch.cuda.memory_allocated()/1024**3:.2f}")
        # if self.debug:
        #     print(f"resized shape: {resized.shape}")
        
        if self.vae:
            return resized, mean, log_var
        else:
            return resized
    
    def set_debug(self, debug):
        """Set the debug mode for the autoencoder

        Args:
            debug (bool): if True, enable debug mode for additional output
        """
        self.debug = debug

def custom_loss(output, target):
    """Calculates a custom reconstruction loss, combining L1 loss for overall structure and MSE loss
    for finer reconstruction details for a balanced spectrogram recosntruction loss.

    Args:
        output (torch.Tensor): model output spectrogram
        target (torch.Tensor): target spectrogram with which to calculate the loss

    Returns:
        float: total_loss, the spectrogram reconstruction loss between the model output and the target
    """
    # L1 loss for overall structure
    l1_loss = nn.L1Loss()(output, target)
    # MSE loss for fine details
    mse_loss = nn.MSELoss()(output, target)
    total_loss = l1_loss + 0.005 * mse_loss
    
    return total_loss

def vae_loss(output, enhanced, mean, logvar):
    """ Calculates combined reconstruction loss and KL divergence between a Normal distribution (0,1)
    and the mean and log variance sampled from the probability distribution for the latent 
    representation when using the VAE option of the model

    Args:
        output (tensor): VAE model output (log mel-spectrogram)
        enhanced (tensor): enhanced target log mel-spectrogram
        mean (tensor): mean tensor of VAE latent representation distribution
        logvar (tensor): log variance tensor of VAE latent representation distribution

    Returns:
        float: VAE loss (reconstruction loss + KL divergence)
    """
    reconstruction_loss = custom_loss(output, enhanced)
    kld = -0.5 * torch.sum(1+logvar - mean.pow(2) - logvar.exp())

    return reconstruction_loss + kld

def train(args, model, device, train_loader, optimizer, epoch, trigger_sync, nbr_columns, name,
           noise_directory, accumulation_steps=4, masking=False, noising=False, enhancer=False):
    """ 
    Train the VariableLengthRAutoencoder model with options for the following tasks: masking
    (i.e. inpainting), noising (i.e. denoising) and enhancing speech log mel-spectrograms. 

    This function handles the training of a VariableLengthRAutoencoder model for a single epoch.
    Depending on the flags 'masking', 'noising' and 'enhancing', the function adapts the training
    process for different tasks. The gradients are accumulated over a specified number of steps
    (accumulation_steps) before performing an optimization step. 

    If the 'enhancer' flag is True, the function trains the model using mid-quality input data and 
    an enhanced version as target data. It supports an optional 'noising' flag, where noise is
    added to the mid-quality input spectrogram, which is denoised and enhanced through the model. 

    If the 'enhancer' flag is False, the function trains the model to reconstruct the original
    spectrogram from a corrupted version through either a 'masking' or 'noising' corruption. The 
    'masking' flag zeroes out a specified number of columns (nbr_columns) from the spectrogram for
    model to be trained for an inpainting task. The 'noising' flag triggers the same denoising task
    as previously stated. 


    Args:
        args (list): list of command line arguments with training configurations
        model (VariableLengthRAutoencoder): model to train
        device (torch.device): device (CPU or GPU) where the model and data are placed
        train_loader (torch.utils.data.DataLoader): data loader object providing training data
        optimizer (optim): optimizer for updating model parameters
        epoch (int): current epoch number for model checkpoints and logging
        trigger_sync (TriggerWandbSyncHook): Weights & Biases offline synchronization trigger
        nbr_columns (int): number of columns to zero out in inpainting task when 'masking' is True
        name (str): current model name, for checkpoints
        noise_directory (str): directory path containing noises for denoising task when 'noising'
                                is True
        accumulation_steps (int, optional): number of steps before accumulating the gradients.
                                            Defaults to 4.
        masking (bool, optional): If True, performs the masking task where random columns are zeroed out
                                in the input data. Defaults to False.
        noising (bool, optional): If True, performs the denoising task where noise is added to the input
                                data. Defaults to False.
        enhancer (bool, optional): If True, performs the enhancing task where parallel mid-quality and
                                enhanced data is used to train the model. Defaults to False.

    Returns:
        None
    
    Notes:
        The training loss and average epoch loss are logged with Weights & Biases
        Model checkpoints are saved after each epoch
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    loss = 0
    if enhancer:
        for batch_idx, (data, enhanced_data, lengths) in enumerate(train_loader):
            # try to change to one loop by unzipping into 2 when enhanced and keeping at one when not enhanced
            data = data.to(device)
            enhanced_data = enhanced_data.to(device)
            lengths = lengths.to(device)
            
            # create mask for padding because of variable length withing 1 batch 
            batch_size, _, height, max_width = data.shape
            mask = torch.arange(max_width, device=device)[None, None, None, :] < lengths[:, None, None, None]
            mask = mask.float()

            if noising:
                noised_tensor = torch.clone(data)
                snr = random.randint(5, 30)
                noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr)
                noised_tensor = noised_tensor.to(device)
                output = model(noised_tensor)
            else:
                output, mean, var = model(data)

            # apply mask to both output and target
            loss = vae_loss(output * mask, enhanced_data * mask, mean, var)
            loss.backward()
            total_loss += loss.item()   

            # perform optimizer step every 'accumulation_steps' batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                wandb.log({"loss": loss.item()})
                trigger_sync()
                subprocess.Popen("sed -i 's|.*|/work/tc062/tc062/s2501147/autoencoder|g' {}/*.command".format(comm_dir),
                                shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)

        # perform a final optimizer step if there are remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print('Train Epoch for model {}: {} Average loss: {:.4f}'.format(name, epoch, avg_loss))
        wandb.log({"epoch_loss": avg_loss})

        torch.save(model.state_dict(), f"{name}/checkpoint_{epoch}.pt")
        print(f"Model saved at epoch {epoch}")

        
    if enhancer==False:
        for batch_idx, (data, lengths) in enumerate(train_loader):
            data = data.to(device)
            lengths = lengths.to(device)
            batch_size, _, height, max_width = data.shape
            
            # create mask for padding
            mask = torch.arange(max_width, device=device)[None, None, None, :] < lengths[:, None, None, None]
            mask = mask.float()

            if masking:
            # zero out random columns in non-padded area
                zeroed_tensor = torch.clone(data)
                for i, length in enumerate(lengths):
                    columns = random.sample(range(length.item()), min(nbr_columns, length.item()))
                    zeroed_tensor[i, :, :, columns] = 0
                output = model(zeroed_tensor)
            
            if noising:
                noised_tensor = torch.clone(data)
                snr = random.randint(5, 30)
                noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr)
                noised_tensor = noised_tensor.to(device)
                output = model(noised_tensor)
            
            # apply mask to both output and target
            loss = custom_loss(output * mask, data * mask)
            loss.backward()
            total_loss += loss.item()

            # perform optimizer step every 'accumulation_steps' batches
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                wandb.log({"loss": loss.item()})
                trigger_sync()
                subprocess.Popen("sed -i 's|.*|/work/tc062/tc062/s2501147/autoencoder|g' {}/*.command".format(comm_dir),
                                shell=True,
                                stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE)

        # perform a final optimizer step if there are remaining gradients
        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print('Train Epoch for model {}: {} Average loss: {:.4f}'.format(name, epoch, avg_loss))
        wandb.log({"epoch_loss": avg_loss})

        torch.save(model.state_dict(), f"{name}/checkpoint_{epoch}.pt")
        print(f"Model saved at epoch {epoch}")

def test(model, device, test_loader, trigger_sync, nbr_columns, noise_directory,
          masking=True, noising=False, enhancer=False):
    """
    Tests a VariableLengthRAutoencoder with options for the following spectrogram corruptions/tasks: 
    'masking' (inpainting task), 'noising' (denoising task) and 'enhancer' (enhancement task)

    This function evaluates the model's performance on a given test dataset using an average test loss,
    provided by the test_loader after training for each epoch. It supports different testing scenarios
    including masking, noising and enhancement of input data. 

    If the 'enhancer' flag is True, it tests the model's ability to enhance the mid-quality input data
    to the enhanced quality of the target. It supports an optional 'noising' flag, where noise is
    added to the mid-quality input spectrogram, which is denoised and enhanced through the model. 

    If the 'enhancer' flag is False, it tests the model's ability to reconstruct the original
    spectrogram from a corrupted version through either a 'masking' or 'noising' corruption. The 
    'masking' flag zeroes out a specified number of columns (nbr_columns) from the spectrogram for
    model to be tested for an inpainting task. The 'noising' flag triggers the same denoising task
    as previously stated. 

    Args:
        model (VariableLengthRAutoencoder): model to be tested
        device (torch.device): device (CPU or GPU) where the model and data are placed
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        trigger_sync (TriggerWandbSyncHook): Weights & Biases offline synchronization trigger
        nbr_columns (int): number of columns to zero out in inpainting task when 'masking' is True
        noise_directory (str): directory path containing noises for denoising task when 'noising'
                                is True        
        masking (bool, optional): If True, performs the masking task where random columns are zeroed out
                                in the input data. Defaults to False.
        noising (bool, optional): If True, performs the denoising task where noise is added to the input
                                data. Defaults to False.
        enhancer (bool, optional): If True, performs the enhancing task where parallel mid-quality and
                                enhanced data is used to train the model. Defaults to False.
    Returns:
        float: average test loss accross all batches in test loader
    """
    model.eval()
    test_loss = 0
    total_pixels = 0
    with torch.no_grad():
        if enhancer:
            for data, enhanced_data, lengths in test_loader:
                data = data.to(device)
                enhanced_data = enhanced_data.to(device)
                lengths = lengths.to(device)
                batch_size, _, height, max_width = data.shape

                # create mask for padding
                mask = torch.arange(max_width, device=device)[None, None, None, :] < lengths[:, None, None, None]
                mask = mask.float()

                if noising:
                    noised_tensor = torch.clone(data)
                    snr = random.randint(10, 30)
                    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr)
                    noised_tensor = noised_tensor.to(device)
                    output = model(noised_tensor)
                else:
                    output, mean, logvar = model(data)

                # apply mask to both output and target
                loss = vae_loss(output * mask, enhanced_data * mask, mean, logvar)
                test_loss += loss.item()


        if enhancer==False:
            for data, lengths in test_loader:
                data = data.to(device)
                lengths = lengths.to(device)
                batch_size, _, height, max_width = data.shape
                
                # create mask for padding
                mask = torch.arange(max_width, device=device)[None, None, None, :] < lengths[:, None, None, None]
                mask = mask.float()

                if masking:
                # zero out random columns (only in non-padded area)
                    zeroed_tensor = torch.clone(data)
                    for i, length in enumerate(lengths):
                        columns = random.sample(range(length.item()), min(nbr_columns, length.item()))
                        zeroed_tensor[i, :, :, columns] = 0
                    output = model(zeroed_tensor)
                
                if noising:
                    noised_tensor = torch.clone(data)
                    snr = random.randint(5, 30)
                    noised_tensor = add_noise_to_spec(noised_tensor, noise_directory, snr)
                    noised_tensor = noised_tensor.to(device)
                    output = model(noised_tensor)

                # apply mask to both output and target
                loss = custom_loss(output * mask, data * mask)
                test_loss += loss.item()
        
    average_test_loss = test_loss / len(test_loader)
    print('\nTest set: Average loss: {:.4f}\n'.format(average_test_loss))
    wandb.log({"test_loss": average_test_loss})
    trigger_sync()
    subprocess.Popen("sed -i 's|.*|/work/tc062/tc062/s2501147/autoencoder|g' {}/*.command".format(comm_dir),
                        shell=True,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE)
        
    return average_test_loss

def enhanced_custom_collate(batch):
    """
    Custom collate function for batching spectrograms and their enhanced versions.

    This function takes a batch of original mid-quality spectrograms, the enhanced target versions
    and their lengths and pads them to an equal length within the batch for processing in the model.
    It ensures that the original and enhanced spectrograms are padded to the same length. 

    Args:
        batch (list): list of tuples, where each tuple contains:
                    (spectrogram, enhance_spectorgram, length)
                    spectrogram (torch.Tensor): the original mid-quality spectrogram 
                    enhanced_spectrogram (torch.Tensor): enhanced version of the spectrogram
                    length (int): original length of the spectrogram (its enhanced version being the
                    same length)

    Returns:
        tuple: tuple containing:
            (padded_spectrograms, padded_enhanced_spectrograms, lengths)
            padded_spectrograms (torch.Tensor): batch of padded original mid-quality spectrograms
            padded_enhanced_spectrograms (torch.Tensor): batch of padded enhanced spectrograms
            lengths (torch.Tensor): original lengths of spectrograms before padding

    Raises:
        AssertionError: if the lengths of the original and enhanced spectrograms don't match

    Note:
        This function assumes that the input spectrogram and enhanced versions are 3D tensors
        (channel, mel_frequency_bins, time), where time is the dimension being padded

    """
    spectrograms, enhanced_spectrograms, lengths = zip(*batch)

    # maximum length in the batch
    max_len = max(spec.shape[2] for spec in spectrograms)

    padded_spectrograms = []
    padded_enhanced_spectrograms = []
    for spec, enhanced_spec in zip(spectrograms, enhanced_spectrograms):
        # ensure both spec and enhanced_spec have the same length
        assert spec.shape[2] == enhanced_spec.shape[2], f"Spectrogram and enhanced spectrogram lengths don't match: {spec.shape} vs {enhanced_spec.shape}"
        
        pad_len = max_len - spec.shape[2]
        padded_spec = F.pad(spec, (0, pad_len), mode='constant', value=0)
        padded_enhanced_spec = F.pad(enhanced_spec, (0, pad_len), mode='constant', value=0)
        
        padded_spectrograms.append(padded_spec)
        padded_enhanced_spectrograms.append(padded_enhanced_spec)

    padded_spectrograms = torch.stack(padded_spectrograms)
    padded_enhanced_spectrograms = torch.stack(padded_enhanced_spectrograms)
    lengths = torch.LongTensor(lengths)

    return padded_spectrograms, padded_enhanced_spectrograms, lengths


def custom_collate(batch):
    """
    Custom collate function for batching spectrograms of variable lengths.

    This function takes a batch of original spectrograms and their lengths and pads them
    to an equal length within the batch for processing in the model. It sorts them in descending
    order of length and pads the spectrograms in the batch to the length of the longest one. 

    Args:
        batch (list): list of tuples, where each tuple contains:
                    (spectrogram, length)
                    spectrogram (torch.Tensor): the original spectrogram 
                    length (int): original length of the spectrogram
    Returns:
        tuple: tuple containing:
            (padded_spectrograms, lengths)
            padded_spectrograms (torch.Tensor): batch of padded original spectrograms
            lengths (torch.Tensor): original lengths of spectrograms before padding
    Note:
        This function assumes that the input spectrogram is a 3D tensor
        (channel, mel_frequency_bins, time), where time is the dimension being padded

    """
    # sort the batch in descending order of length
    batch.sort(key=lambda x: x[1], reverse=True)
    spectrograms, lengths = zip(*batch)

    # pad spectrograms to the length of longest 
    max_len = max(lengths)
    padded_spectrograms = []
    for spec in spectrograms:
        pad_len = max_len - spec.shape[2]  # padding on time dim
        padded_spec = np.pad(spec, ((0, 0), (0, 0), (0, pad_len)), mode='constant')
        padded_spectrograms.append(padded_spec)

    # convert to tensors
    padded_spectrograms = torch.FloatTensor(np.stack(padded_spectrograms))
    lengths = torch.LongTensor(lengths)

    return padded_spectrograms, lengths


def main():
    # training configuration
    parser = argparse.ArgumentParser(description='PyTorch Autoencoder for Sepctrograms')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=12, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--dev-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    print("device: ", device)
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    dev_kwargs = {'batch_size': args.dev_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        dev_kwargs.update(cuda_kwargs)
        
    base_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset"
    
    train_dataset, dev_dataset, test_dataset = load_datasets(base_dir)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs, collate_fn=custom_collate, shuffle=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs, collate_fn=custom_collate, shuffle=None)


    model = VariableLengthRAutoencoder(vae=False).to(device)
    # model.load_state_dict(torch.load("enhancer_finetuned/checkpoint_4.pt"))
    # model.to(device)

    mask = 5
    fine_tune_mask = 10

    model_name = "denoiser_aircon_skip_connections"
    train_noise_dir = "/work/tc062/tc062/s2501147/autoencoder/noise_dataset/mels/only_aircon"
    test_noise_dir = "/work/tc062/tc062/s2501147/autoencoder/noise_dataset/mels/only_aircon_test"
    # wandb
    wandb.init(config=args, dir="/work/tc062/tc062/s2501147/autoencoder", mode="offline")
    wandb.watch(model, log_freq=100)
    trigger_sync = TriggerWandbSyncHook(communication_dir = comm_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, trigger_sync,
                            mask, model_name, train_noise_dir,
                            masking=False, noising=True, enhancer=False)
        test_loss = test(model, device, test_loader, trigger_sync, mask,
                            test_noise_dir, masking=False, noising=True, enhancer=False)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"{model_name}.pt")

if __name__ == '__main__':
    print("working directory: ", os.getcwd())
    # os.chdir("/work/tc062/tc062/s2501147/autoencoder")
    # print("working directory: ", os.getcwd())
    main()
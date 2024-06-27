import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
import random
import numpy as np
from spec_dataset import SpectrogramDataset, create_dataloaders, load_datasets
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  

comm_dir = "/work/tc062/tc062/s2501147/autoencoder/.wandb_osh_command_dir"

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, embedding_dim, lstm_units, hidden_dim, num_classes, lstms_layers, 
                        bidirectional, droupout, pad_index, batch_size, cnn):

        super(LSTMAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
           # TODO try 2 biLSTM layers + 1 conv + 1 linear
        )


        # Decoder
        self.decoder = nn.Sequential(
            # TODO try 2 biLSTM layers + 1 conv + 1 linear
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch, trigger_sync, save_dir, nbr_columns):
    model.train()
    distance = nn.MSELoss()
    total_loss = 0
    counter = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
        # make directly 1 column zeroed out for this batch 
        columns = random.sample(range(0, 79), nbr_columns)
        # print("column: ", column)
        zeroed_tensor = torch.clone(data)
        # print("size of zeroed_tensor: ", zeroed_tensor.size())
        for column in columns:
            zeroed_tensor[:,:, :, column]=0
        # save tensor and print tensor; needs to be on cpu
        cpu_z_tensor = zeroed_tensor.to('cpu')
        numpy_zero_tensor = cpu_z_tensor.numpy()
        np.save(f"{save_dir}/zeroed_numpy_tensor_{counter}.npy", numpy_zero_tensor)
        counter +=1 
        # print("size of zeroed_tensor: ", zeroed_tensor.size())
        
        optimizer.zero_grad()
        output = model(zeroed_tensor)
        # print("size of output: ", output.size())
        # print("size of data: ", data.size())
        
        loss = distance(output, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
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
    
    avg_loss = total_loss / len(train_loader)
    print('Train Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))
    wandb.log({"epoch_loss": avg_loss})

def test(model, device, test_loader, trigger_sync, nbr_columns):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # make directly 1 column zeroed out for this batch 
            columns = random.sample(range(0,79), nbr_columns)
            zeroed_tensor = torch.clone(data)
            for column in columns:
                zeroed_tensor[:,:,:, column]=0

            output = model(zeroed_tensor)
            test_loss += F.mse_loss(output, data, reduction='sum').item()
    
    test_loss /= len(test_loader.dataset) * 80 * 80  # Average loss per pixel
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    wandb.log({"test_loss": test_loss})
    trigger_sync()
    subprocess.Popen("sed -i 's|.*|/work/tc062/tc062/s2501147/autoencoder|g' {}/*.command".format(comm_dir),
                                    shell=True,
                                    stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE)
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Autoencoder for Sepctrograms')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--val-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for validation (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
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
    val_kwargs = {'batch_size': args.val_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)
        
    base_dir = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays"
    
    train_dataset, val_dataset, test_dataset = load_datasets(base_dir)
    # train_loader, val_loader, test_loader = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=64)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_dataset = torch.utils.data.DataLoader(val_dataset, **val_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # gold_train_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/train"
    # gold_train_dataset = SpectrogramDataset(gold_train_data)
    # gold_train_loader = torch.utils.data.DataLoader(gold_train_dataset, **train_kwargs)
    
    # gold_test_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/test"
    # gold_test_dataset = SpectrogramDataset(gold_train_data)
    # gold_test_loader = torch.utils.data.DataLoader(gold_test_dataset, **test_kwargs)

    # gold_val_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val"
    # gold_val_dataset = SpectrogramDataset(gold_val_data)
    # gold_val_loader = torch.utils.data.DataLoader(gold_val_dataset, **val_kwargs)


    model = RAutoencoder().to(device)

    test_dir = "/work/tc062/tc062/s2501147/autoencoder/test_zeroed_tensors"
    mask = 1

    # wandb
    wandb.init(config=args, dir="/work/tc062/tc062/s2501147/autoencoder", mode="offline")
    wandb.watch(model, log_freq=100)
    trigger_sync = TriggerWandbSyncHook(communication_dir = comm_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, trigger_sync, test_dir, mask)
        test_loss = test(model, device, test_loader, trigger_sync, mask)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "reconstructor_1024_3layers_4paddings.pt")





if __name__ == '__main__':
    print("working directory: ", os.getcwd())
    # os.chdir("/work/tc062/tc062/s2501147/autoencoder")
    # print("working directory: ", os.getcwd())
    main()
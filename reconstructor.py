import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess
from spec_dataset import SpectrogramDataset
from recon_dataset import ZeroedSpectrogramDataset, create_dataloaders, load_datasets
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook  

comm_dir = "/work/tc062/tc062/s2501147/autoencoder/.wandb_osh_command_dir"

class RAutoencoder(nn.Module):
    def __init__(self):

        super(RAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # output size: (32, 40, 40)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # output size: (64, 20, 20)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # output size: (128, 10, 10)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # output size: (256, 5, 5)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # output size: (512, 5, 5)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),  # output size: (1024, 5, 5)
            # nn.BatchNorm2d(1024),
            # nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1),  # output size: (512, 5, 5)
            # nn.BatchNorm2d(512),
            # nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # output size: (256, 5, 5)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # output size: (128, 10, 10)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # output size: (64, 20, 20)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # output size: (32, 40, 40)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # output size: (1, 80, 80)
            # nn.Sigmoid()  # To output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# def train(args, model, device, train_loader, gold_standard_loader, optimizer, epoch, trigger_sync):
#     model.train()
#     distance = nn.MSELoss()
#     for batch_idx, (data, gold_standard) in enumerate(zip(train_loader, gold_standard_loader)):
#         data, gold_standard = Variable(data).to(device), Variable(gold_standard).to(device)
#         output = model(data)
#         print("data size: ", output.size())
#         print("gold standard: ", gold_standard.size())
#         loss = distance(output, gold_standard)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if batch_idx % args.log_interval == 0:
#             wandb.log({"loss": loss})
#             trigger_sync()
#             subprocess.Popen("sed -i 's|.*|/work/tc062/tc062/s2501147/autoencoder|g' {}/*.command".format(comm_dir),
#                                                 shell=True,
#                                                 stdout=subprocess.PIPE,
#                                                 stdin=subprocess.PIPE)
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
#             if args.dry_run:
#                 break

def train(args, model, device, train_loader, gold_standard_loader, optimizer, epoch, trigger_sync):
    model.train()
    distance = nn.MSELoss()
    total_loss = 0
    for batch_idx, (data, gold_standard) in enumerate(zip(train_loader, gold_standard_loader)):
        data, gold_standard = data.to(device), gold_standard.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = distance(output, gold_standard)
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

def test(model, device, test_loader, gold_standard_loader, trigger_sync):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, gold_standard in zip(test_loader, gold_standard_loader):
            data = data.to(device)
            gold_standard = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, gold_standard, reduction='sum').item()
    
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
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    # will not work, need to import other dataset
    gold_train_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/train"
    gold_train_dataset = SpectrogramDataset(gold_train_data)
    gold_train_loader = torch.utils.data.DataLoader(gold_train_dataset, **train_kwargs)
    
    gold_test_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/test"
    gold_test_dataset = SpectrogramDataset(gold_train_data)
    gold_test_loader = torch.utils.data.DataLoader(gold_test_dataset, **test_kwargs)

    gold_val_data = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_arrays/val"
    gold_val_dataset = SpectrogramDataset(gold_val_data)
    gold_val_loader = torch.utils.data.DataLoader(gold_val_dataset, **val_kwargs)


    model = RAutoencoder().to(device)

    # wandb
    wandb.init(config=args, dir="/work/tc062/tc062/s2501147/autoencoder", mode="offline")
    wandb.watch(model, log_freq=100)
    trigger_sync = TriggerWandbSyncHook(communication_dir = comm_dir)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, gold_train_loader, optimizer, epoch, trigger_sync)
        test_loss = test(model, device, test_loader, gold_test_loader, trigger_sync)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "reconstructor_all_layers.pt")





if __name__ == '__main__':
    print("working directory: ", os.getcwd())
    # os.chdir("/work/tc062/tc062/s2501147/autoencoder")
    # print("working directory: ", os.getcwd())
    main()
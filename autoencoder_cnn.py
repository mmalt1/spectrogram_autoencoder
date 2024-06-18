
import argparse
import torch
import os
# import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  # Updated to 64*12*12
        self.fc2 = nn.Linear(128, 64)

        # Decoder
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64 * 12 * 12)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.25)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 1)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 3, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Encoding
        # size of x = [64, 1, 28, 28]
        x = self.conv1(x)           # size = [64, 32, 26, 26]
        x = F.relu(x)               # size = [64, 32, 26, 26]
        x = self.conv2(x)           # size = [64, 64, 24, 24]
        x = F.relu(x)               # size = [64, 64, 24, 24]
        x = F.max_pool2d(x, 2)      # size = [64, 64, 12, 12]
        x = self.dropout1(x)        # size = [64, 64, 12, 12]
        x = torch.flatten(x, 1)     # size = [64, 9216]
        x = self.fc1(x)             # size = [64, 128]
        x = F.relu(x)               # size = [64, 128]
        x = self.dropout2(x)        # size = [64, 128]
        output = self.fc2(x)        # size = [64, 64]
        # print("size of output: ", output.size())
        # output = F.log_softmax(x, dim=1)  # size = [64, 64]

        # Decoding
        y = self.fc3(output)        # size = [64, 128]
        # print("size of y: ", y.size())
        y = self.dropout3(y)        # size = [64, 128]
        y = F.relu(y)               # size = [64, 128]
        y = self.fc4(y)             # size = [64, 9216]
        # y = torch.unflatten(y, 1,(64, 12, 12)) # size = [64, 64, 12, 12]
        y = y.view(-1, 64, 12, 12)  # [64, 64, 12, 12]
        y = F.relu(y)               # size = [64, 64, 12, 12]
        y = self.dropout4(y)        # size = [64, 64, 12, 12]
        y = F.relu(y)               # size = [64, 64, 12, 12]
        y = self.deconv1(y)         # size = [64, 32, 14, 14]
        y = F.relu(y)               # size = [64, 32, 14, 14]
        y = self.deconv2(y)         # size = [64, 1, 16, 16]
        final = F.interpolate(y, size=(28, 28), mode='nearest')  # size = [64, 1, 28, 28]
        final = self.sig(final)         # size = [64, 1, 28, 28]


        return final


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    distance = nn.MSELoss()
    # train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data).to(device)
        output = model(data)
        loss = distance(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print('epoch [{}], loss: {:.4f}'.format(epoch, loss.item()))
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data, reduction='sum').item()
    
    test_loss /= len(test_loader.dataset) * 28 * 28  # Average loss per pixel
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Autoencoder Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
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
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_dir = '/work/tc062/tc062/s2501147/autoencoder/preloaded_data'
    dataset1 = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST(root=data_dir, train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "autoenc_10epochs.pt")


if __name__ == '__main__':
    print("working directory: ", os.getcwd())
    os.chdir("/work/tc062/tc062/s2501147/autoencoder")
    print("working directory: ", os.getcwd())
    main()
       


"""
After 14 epochs?? Maybe?? 

"""
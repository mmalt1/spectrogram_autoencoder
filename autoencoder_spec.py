import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spec_dataset import SpectrogramDataset, create_dataloaders, load_datasets
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable


class Autoencoder(nn.Module):
    def __init__(self):

        super(Autoencoder, self).__init__()

        # Encoder

        #change the stride and look at the difference
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1, output_padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1, output_padding=1), 
            nn.Sigmoid()  # To output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    #     super(Autoencoder, self).__init__()
    #     # Encoder
    #     self.conv1 = nn.Conv2d(1, 16, 1, 1)
    #     # self.fc1 = nn.Linear(80, 80, False)
    #     # self.conv1 = nn.Conv2d(1, 32,  1, 1)
    #     self.conv2 = nn.Conv2d(16, 32, 1, 1)
    #     self.conv3 = nn.Conv2d(32, 64, 1, 1)
    #     self.conv4 = nn.Conv2d(64, 128, 1, 1)
    #     self.dropout1 = nn.Dropout(0.25)
    #     self.dropout2 = nn.Dropout(0.5)
    #     # self.fc1 = nn.Linear(64*80*80, 512)  # Updated to 64*38*38
    #     # self.fc2 = nn.Linear(512, 64)
    #     self.fc1 = nn.Linear(80, 80)
    #     self.fc2 = nn.Linear(80, 80) #maybe try 1024 (according to Simon)

    #     # Decoder
    #     # self.deconv1 = nn.ConvTranspose2d(64, 32, 1, 1)
    #     # self.fc2 = nn.Linear(80, 80, False)
    #     self.fc3 = nn.Linear(80, 80)
    #     self.fc4 = nn.Linear(80, 80)
    #     self.dropout3 = nn.Dropout(0.5)
    #     self.dropout4 = nn.Dropout(0.25)
    #     self.deconv4 = nn.ConvTranspose2d(128, 64, 1, 1)
    #     self.deconv3 = nn.ConvTranspose2d(64, 32, 1, 1)
    #     self.deconv1 = nn.ConvTranspose2d(32, 16, 1, 1)
    #     self.deconv2 = nn.ConvTranspose2d(16, 1, 1, 1)
    #     self.sig = nn.Sigmoid()

    # def forward(self, x):
    #     # Encoding
    #     # print("size of input: ", x.size())
    #     # x = self.fc1(x)
    #     # x = x
    #     # print("size x: ", x.size())


    #     # print("size of x input: ", x.size())
    #     x = self.conv1(x)           # size = [batch_size, 32, 78, 78]
    #     # print("size of x after conv1: ", x.size())
    #     x = F.relu(x)               # size = [batch_size, 32, 78, 78]
    #     x = self.conv2(x)           # size = [batch_size, 64, 76, 76]
    #     # print("size of x after conv2: ", x.size())
    #     x = F.relu(x)               # size = [batch_size, 64, 76, 76]
    #     x = self.conv3(x)
    #     x = F.relu(x)
    #     # x = F.max_pool2d(x, 2)      # size = [batch_size, 64, 38, 38]
    #     # print("size of x after relu1: ", x.size())
    #     # y = self.dropout1(x)        # size = [batch_size, 64, 38, 38]
    #     # x = torch.flatten(x, 1)     # size = [batch_size, 64*38*38]
    #     # print("size of x after flatten: ", x.size())
    #     x = self.conv4(x)
    #     # encoded = self.fc1(x)             # size = [batch_size, 128]
    #     # print("size of x after fc1: ", x.size())
    #     x = F.relu(x)               # size = [batch_size, 128]
    #     # x = self.dropout2(x)        # size = [batch_size, 128]
    #     encoded = self.fc2(x)       # size = [batch_size, 64]
    #     # print("size of x after fc2: ", encoded.size())

    #     # Decoding
    #     y = self.fc3(encoded)       # size = [batch_size, 128]
    #     # print("size of x after fc3: ", y.size())
    #     # y = self.dropout3(y)        # size = [batch_size, 128]
    #     y = F.relu(y)               # size = [batch_size, 128]
    #     # y = self.fc4(encoded)             # size = [batch_size, 64*38*38]
    #     # print("size of y after fc4: ", y.size())
    #     y = self.deconv4(y)
    #     y = F.relu(y)
    #     # y = y.view(-1, 64, 80, 80)  # size = [batch_size, 64, 38, 38]
    #     y = F.relu(y)               # size = [batch_size, 64, 38, 38]
    #     y = self.deconv3(y)
    #     # y = self.dropout4(y)        # size = [batch_size, 64, 38, 38]
    #     y = F.relu(y)               # size = [batch_size, 64, 38, 38]
    #     y = self.deconv1(y)         # size = [batch_size, 32, 40, 40]
    #     # print("size of final y after deconv1: ", y.size())
    #     y = F.relu(y)               # size = [batch_size, 32, 40, 40]
    #     final = self.deconv2(y)         # size = [batch_size, 1, 42, 42]
    #     # print("size of final after deconv2: ", y.size())
    #     # final = F.interpolate(y, size=(80, 80), mode='nearest')  # size = [batch_size, 1, 80, 80]
    #     final = self.sig(final)     # size = [batch_size, 1, 80, 80]


    #     return final


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    distance = nn.MSELoss()
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data).to(device)
        output = model(data)
        loss = distance(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data, reduction='sum').item()
    
    test_loss /= len(test_loader.dataset) * 80 * 80  # Average loss per pixel
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
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
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
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
    
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "spec_stride1_autoencoder.pt")


if __name__ == '__main__':
    print("working directory: ", os.getcwd())
    os.chdir("/work/tc062/tc062/s2501147/autoencoder")
    print("working directory: ", os.getcwd())
    main()
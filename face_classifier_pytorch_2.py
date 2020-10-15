from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
from PIL import Image
import os


class FaceMaskDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # image = cv2.imread(img_path)
        image = Image.open(img_path)
        y_label = torch.tensor(int(0 if self.annotations.iloc[index, 1] == "No-Mask" else 1))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        # self.fc2 = nn.Linear(9216, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x, v=False):
        x = self.conv1(x)
        # print("1", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        # print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print("3", x.shape)
        x = self.conv3(x)
        # print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv4(x)
        # print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print("4", x.shape)
        x = torch.flatten(x, 1)
        # print("5", x.shape)
        x = self.fc1(x)
        # print("6", x.shape)
        x = F.relu(x)
        x = self.dropout2(x)
        # print("7", x.shape)
        # x = self.fc2(x)
        # x = F.relu(x)
        x = self.dropout2(x)
        # print("7", x.shape)
        x = self.fc3(x)
        # print("8", x.shape)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss = criterion(output, target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    #     ])

    transforms_fn = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor()])

    # Load Data
    data_root_dir = "/home/ambolt/Data/emily/faces/"
    label_csv = os.path.join(data_root_dir, "labels.csv")
    img_dir = os.path.join(data_root_dir, "images_big")
    dataset = FaceMaskDataset(csv_file=label_csv, root_dir=img_dir, transform=transforms_fn)

    train_samples = int(round(len(dataset) * 0.8, 0))
    test_samples = int(round(len(dataset) * 0.2, 0))
    train_set, test_set = torch.utils.data.random_split(dataset, [train_samples, test_samples])


    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)

    model = Net().to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, criterion, test_loader)
        scheduler.step()

        if args.save_model:
            torch.save(model, f"./models/custom{str(epoch + 1)}.pth")


if __name__ == '__main__':
    main()
from __future__ import print_function
import argparse
import os

import torch
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from model import CustomModel, ResNetModel
from facemask_dataset import FaceMaskDataset
from transforms import Transformer


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
    torch.no_grad()

    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = criterion(output, target)
        # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Mask Classifier Example')
    parser.add_argument('--train-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Load Data
    data_root_dir = "/home/ambolt/Data/emily/faces/"
    img_dir = os.path.join(data_root_dir, "images")
    assert os.path.exists(img_dir) and len(os.listdir(img_dir)) > 100, "Couldn't find the specified path to images"

    train__csv = os.path.join(data_root_dir, "train.csv")
    test_csv = os.path.join(data_root_dir, "test.csv")

    train_dataset = FaceMaskDataset(csv_file=train__csv, img_dir=img_dir, transform=Transformer().train_transforms())
    test_dataset = FaceMaskDataset(csv_file=test_csv, img_dir=img_dir, transform=Transformer().test_transforms())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch_size,
                                               shuffle=True,
                                               num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=4)

    model = ResNetModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, criterion, test_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            model.save_model(f"./models/2021-01-28/resnet_model_epoch_{epoch + 1}_acc_{round(acc, 2)}.pth")


if __name__ == '__main__':
    main()

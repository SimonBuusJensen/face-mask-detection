import os
import argparse
import torch
from torch.utils.data import DataLoader

from facemask_dataset import FaceMaskDataset
from transforms import Transformer
from model import ResNetModel


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
    torch.no_grad()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


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
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    # Load Data
    img_dir = "/home/ambolt/Data/emily/faces/images"
    train_csv = os.path.join(os.path.dirname(img_dir), "train.csv")
    test_csv = os.path.join(os.path.dirname(img_dir), "test.csv")

    transformer = Transformer()
    train_dataset = FaceMaskDataset(csv_file=train_csv, img_dir=img_dir, transform=transformer.train_transforms())
    test_dataset = FaceMaskDataset(csv_file=test_csv, img_dir=img_dir, transform=transformer.test_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True)

    model = ResNetModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, epoch)
        acc = test(model, device, criterion, test_loader)
        scheduler.step()

        if acc > best_acc:
            acc = best_acc
            model.save_model(f"./models/2020-10-19_custom_gray_epoch_{str(epoch + 1)}_acc_{acc}.pth")


if __name__ == '__main__':
    main()

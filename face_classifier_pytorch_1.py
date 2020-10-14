# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
import torchvision
import os
import pandas as pd
from skimage import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader  # Gives easier dataset managment and creates mini batches


class FaceMaskDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel = 3
num_classes = 2
learning_rate = 1e-3
batch_size = 32
num_epochs = 10

# Load Data
data_root_dir = "/home/ambolt/Data/emily/MAFA_faces/train/"
label_csv = os.path.join(data_root_dir, "labels.csv")
img_dir = os.path.join(data_root_dir, "images")
transforms_fn = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])
dataset = FaceMaskDataset(csv_file=label_csv, root_dir=img_dir, transform=transforms_fn)

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
train_samples = int(round(len(dataset) * 0.8, 0))
test_samples = int(round(len(dataset) * 0.2, 0))
train_set, test_set = torch.utils.data.random_split(dataset, [train_samples, test_samples])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
# model = torchvision.models.googlenet(pretrained=True)
model = torchvision.models.resnet50(pretrained=True)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    torch.save(model, f"./models/resnet50_epoch_{str(epoch + 1)}.pth")
    print("Checking accuracy on Test Set")
    check_accuracy(test_loader, model)
    print(f'Loss at epoch {epoch + 1} is {sum(losses) / len(losses)}')

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)



# Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn.functional as F
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
        y_label = torch.tensor(int(0 if self.annotations.iloc[index, 1] == "No-Mask" else 1))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

# define the CNN architecture
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
       self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
       self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
       self.pool = nn.MaxPool2d(2, 2)
       self.fc1 = nn.Linear(64*4*4*4*4, 64)
       self.fc2 = nn.Linear(64, 1)
       self.dropout = nn.Dropout(0.1)

   def forward(self, x):
       # add sequence of convolutional and max pooling layers
       print("1", x.shape)
       x = self.pool(F.relu(self.conv1(x)))
       print("2", x.shape)
       x = self.pool(F.relu(self.conv2(x)))
       print("3", x.shape)
       x = self.pool(F.relu(self.conv3(x)))
       print("4", x.shape)
       x = x.view(-1, 32 * 4 * 4*4*4)
       print("5", x.shape)
       x = self.dropout(x)
       print("6", x.shape)
       x = F.relu(self.fc1(x))
       print("7", x.shape)
       x = self.dropout(x)
       print("8", x.shape)
       x = F.relu(self.fc2(x))
       print("9", x.shape)
       x = F.sigmoid(x)
       print("10", x.shape)
       return x






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
learning_rate = 1e-3
batch_size = 32
num_epochs = 12

# Load Data
data_root_dir = "/home/ambolt/Data/emily/faces/"
label_csv = os.path.join(data_root_dir, "labels.csv")
img_dir = os.path.join(data_root_dir, "images_big")
transforms_fn = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                    ])
dataset = FaceMaskDataset(csv_file=label_csv, root_dir=img_dir, transform=transforms_fn)

# Dataset is actually a lot larger ~25k images, just took out 10 pictures
# to upload to Github. It's enough to understand the structure and scale
# if you got more images.
# train_samples = int(round(len(dataset) * 0.8, 0))
# train_samples = int(round(len(dataset) * 0.8, 0))
# test_samples = int(round(len(dataset) * 0.2, 0))
# train_set, test_set = torch.utils.data.random_split(dataset, [train_samples, test_samples])
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# test_loader = DataLoader(dataset=test_set, batch_size=batch_size)

# Model
model = torchvision.models.googlenet(pretrained=True)
# model = Net()
# print(model)
# model = torchvision.models.resnet50(pretrained=True)
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

    torch.save(model, f"./models/googlenet_{str(epoch + 1)}.pth")
    print("Checking accuracy on Test Set")
    # check_accuracy(test_loader, model)
    print(f'Loss at epoch {epoch + 1} is {sum(losses) / len(losses)}')

print("Checking accuracy on Training Set")
check_accuracy(train_loader, model)



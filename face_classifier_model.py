import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from utils.transforms import *


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
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, verbose=False):
        x = self.conv1(x)
        if verbose:
            print("1", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        if verbose:
            print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if verbose:
            print("3", x.shape)
        x = self.conv3(x)
        if verbose:
            print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv4(x)
        if verbose:
            print("2", x.shape)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        if verbose:
            print("4", x.shape)
        x = torch.flatten(x, 1)
        if verbose:
            print("5", x.shape)
        x = self.fc1(x)
        if verbose:
            print("6", x.shape)
        x = F.relu(x)
        x = self.dropout2(x)
        if verbose:
            print("7", x.shape)
        x = self.dropout2(x)
        if verbose:
            print("7", x.shape)
        x = self.fc2(x)
        if verbose:
            print("8", x.shape)
        output = F.log_softmax(x, dim=1)
        return output


def inference(image, model):
    transforms_fn = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.ToTensor()])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        image = transforms_fn(image)
        image = image.unsqueeze(0)
        image = image.to(device=device)
        score = model(image)
        _, prediction = score.max(1)
        model.train()
        return prediction.detach().cpu().numpy()[0]


if __name__ == '__main__':
    image_path = "/home/ambolt/Data/emily/faces/images_big/train_00001269.jpg"
    image = Image.open(image_path)

    model = torch.load("/home/ambolt/Ambolt/emily/emily-face-mask-detection/models/custom14.pth")

    pred = inference(image, model)
    #
    print(pred)



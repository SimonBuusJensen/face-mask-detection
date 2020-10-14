import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from utils.transforms import *


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def inference(image, model):
    transforms_fn = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
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
    # image_path = "/home/ambolt/Data/emily/MAFA_faces/train/images/test_00000003.jpg"
    # image = Image.open(image_path)

    model = torch.load("/home/simon/projects/emily-face-mask-detection/models/google_LeNet_epoch_10.pth")

    # pred = inference(image, model)
    #
    # print(pred)



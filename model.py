import pickle as pkl

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils.transforms import *


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4608, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc3(x)
        return output

    def save_model(self, save_path):
        """
        Implement steps for saving the model instance.
        e.g. using the pickle module (pickle.dump(model)) for sklearn models or torch.save(model) for PyTorch models
        save_path : str
            The path where a trained model should be saved to
        """
        torch.save(self.state_dict(), save_path)

    def load_model(self, model_path):
        """
        Implement steps for loading a trained model
        e.g. using the pickle module (pickle.load(model)) for sklearn models or torch.load for PyTorch models'
        model_path : str
            The path where the trained model is located and should be loaded from
        """
        self.load_state_dict(torch.load(model_path))

    def __call__(self, sample):
        return self.forward(sample)


class ResNetModel(nn.Module):

    def __init__(self):
        super().__init__()  # Inherit methods from the super class which this class extends from
        self.resnet = models.resnet18(pretrained=True, progress=False)

        # Disable gradients for all convolutional layers in resnet
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 256)
        self.act = torch.nn.ReLU()
        self.fc = torch.nn.Linear(256, 128)
        self.act1 = torch.nn.ReLU()
        # Final layer will output 2 values
        self.fc1 = torch.nn.Linear(128, 2)

    def forward(self, sample):
        """
        Implement the steps which forwards the inputted sample into the model
        e.g. super().__call__(), self.predict(), self.predict_proba etc.
        The implementation depends on the Machine Learning model this Model class extends
        """
        X = self.resnet(sample)
        X = self.act(X)
        X = self.fc(X)
        X = self.act1(X)
        output = self.fc1(X)
        return output

    def save_model(self, save_path):
        """
        Implement steps for saving the model instance.
        e.g. using the pickle module (pickle.dump(model)) for sklearn models or torch.save(model) for PyTorch models
        save_path : str
            The path where a trained model should be saved to
        """
        torch.save(self.state_dict(), save_path)

    def load_model(self, model_path):
        """
        Implement steps for loading a trained model
        e.g. using the pickle module (pickle.load(model)) for sklearn models or torch.load for PyTorch models'
        model_path : str
            The path where the trained model is located and should be loaded from
        """
        self.load_state_dict(torch.load(model_path))

    def __call__(self, sample):
        return self.forward(sample)

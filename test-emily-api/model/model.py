import numpy as np
import torch
from torch import nn 
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
import base64


class Model(): 

    def __init__(self):
        pass

    def predict(self, sample): 
        
        # Create or load model
        self.model = self.get_model()

        print(sample[:10])

        # Preprocessing
        sample = self.preprocess(sample)

        # Prediction
        self.model.eval()
        torch.no_grad()
        score = self.model(sample)
        print(score)
        
        # Postprocessing
        prediction = self.postprocess(score)

        return prediction

    def preprocess(self, x, device="cpu"):

        im = Image.open(BytesIO(base64.b64decode(x)))
    
        transform_fns = transforms.Compose(
            [
                transforms.Resize((128, 128)), 
                transforms.ToTensor()
            ])

        pilimage_tensor = transform_fns(im)
        pilimage_tensor = pilimage_tensor.unsqueeze(0)

        # Prepare for the CPU or GPU 
        pilimage_tensor = pilimage_tensor.to(device=device)
        return pilimage_tensor

    def postprocess(self, x):
        _, prediction = x.max(1)
        prediction = prediction.detach().cpu().numpy()[0]
        return prediction

    def get_model(self):
        # Method for instantiating or loading a model
        model = Net()
        return model 



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
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

        


import torch
from torchvision import transforms

from PIL import Image
from io import BytesIO

import base64

from FaceDetector import FaceDetector


class Model:

    def __init__(self, model_path=None):

        # Load the model
        if model_path:
            self._model = torch.load(model_path, map_location=torch.device('cpu'))
            self.is_fitted = True
        else:
            self._model = Net()
            self.is_fitted = False

        # Set the model in evaluation mode
        self._model.eval()

        self._face_detector = FaceDetector()

    def predict(self, sample):

        if self.is_fitted:
            # Preprocess input
            image = self._preprocess(sample)

            # Forward input into the model
            score = self._forward(image)

            # Postprocessing
            prediction = self._postprocess(score)

            return prediction

        else:
            raise NotImplementedError("WARNING: Model has not been fitted. Fit a model before running predict()")

    def _forward(self, sample):

        # Disabling gradient calculation to reduce memory consumption for computations
        torch.no_grad()

        # Forward the sample into the model
        score = self._model(sample)
        return score

    def _preprocess(self, sample, device="cpu"):

        # Convert the base64 encoded image into a PIL Image
        self._image = Image.open(BytesIO(base64.b64decode(sample)))

        # Extract the face from the image
        self._face = self._face_detector.detect_face(self._image)
        (x1, y1, x2, y2) = self._face
        self._face_image = self._image[y1:y2, x1:x2].copy()

        # Define transformation functions and compose into a single PyTorch transformation function
        transform_fns = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        # Transform image into a PyTorch Tensor
        tensor = transform_fns(self._face_image)

        # Append an extra dimension to the image (128, 128, 3) -> (1, 128, 128, 3)
        tensor = tensor.unsqueeze(0)

        # Prepare the sample for the CPU or GPU
        tensor = tensor.to(device=device)
        return tensor

    def _postprocess(self, sample):
        _, prediction = sample.max(1)
        prediction = prediction.detach().cpu().numpy()[0]
        return prediction


class Net(torch.nn.Module):
    """
    A simple CNN Model for binary classification implemented in PyTorch
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, 1)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, 1)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(4608, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

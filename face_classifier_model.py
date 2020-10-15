import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from utils.transforms import *


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
    # image_path = "/home/ambolt/Data/emily/MAFA_faces/train/images/test_00000003.jpg"
    # image = Image.open(image_path)

    model = torch.load("/home/simon/projects/emily-face-mask-detection/models/google_LeNet_epoch_10.pth")

    # pred = inference(image, model)
    #
    # print(pred)



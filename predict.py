import torch
import os

from PIL import Image

from transforms import Transformer
from model import CustomModel, ResNetModel

class_names = ["No-mask", "Mask"]

def inference(model: torch.nn.Module, image, transforms):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        image = transforms(image)
        image = image.unsqueeze(0)
        image = image.to(device=device)
        score = model.forward(image)

        _, pred_class = score.max(1)
        pred_class = pred_class.detach().cpu().numpy()[0]

        softmax_score = torch.nn.functional.softmax(score, dim=1)
        pred_conf = softmax_score.detach().cpu().numpy()[0][pred_class]

        return pred_class, pred_conf


# if __name__ == '__main__':
#
#     resnet_model_path = 'models/2021-01-28/resnet_model_epoch_5_acc_97.37.pth'
#     model_path = 'models/2021-01-28/custom_model_epoch_8_acc_95.3.pth'
#     img_dir = 'examples'
#
#     # Set cuda settings
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     # Instantiate model
#     resnet_model = ResNetModel().to(device)
#     custom_model = CustomModel().to(device)
#
#     resnet_model.load_model(resnet_model_path)
#     custom_model.load_model(model_path)
#
#     transformer = Transformer()
#
#     for img_name in os.listdir(img_dir):
#         img = Image.open(os.path.join(img_dir, img_name))
#
#         pred_class, pred_conf = inference(resnet_model, img, transformer.test_transforms())
#         print(f"ResNet model predicted {class_names[pred_class]} with {pred_conf} confidence for image", img_name)
#
#         pred_class, pred_conf = inference(custom_model, img, transformer.test_transforms())
#         print(f"Custom model predicted {class_names[pred_class]} with {pred_conf} confidence for image", img_name)
#
#         print()

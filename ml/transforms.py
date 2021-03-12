from torchvision import transforms

RESIZE_W = 256
RESIZE_H = 256


# For training we will do random image augmentation methods such as color jitter and horizontal flip.
# See examples at examples/augmentation
def train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.Resize(size=(RESIZE_H, RESIZE_W)),
        transforms.ToTensor()
    ])


# For test and validation we will solely resize the image and convert
# it to a tensor to prepare it for the torch model
def test_transforms():
    return transforms.Compose([
        transforms.Resize(size=(RESIZE_H, RESIZE_W)),
        transforms.ToTensor()
    ])


def validation_transforms():
    return test_transforms()

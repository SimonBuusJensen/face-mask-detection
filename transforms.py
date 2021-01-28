from torchvision import transforms


class Transformer:

    def train_transforms(self):
        return transforms.Compose([transforms.Resize((128, 128)),
                                   transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
                                   transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
                                   transforms.RandomHorizontalFlip(0.5),
                                   transforms.ToTensor()])

    def test_transforms(self):
        return transforms.Compose([transforms.Resize((128, 128)),
                                   transforms.ToTensor()])

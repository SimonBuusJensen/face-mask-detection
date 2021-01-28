import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image


class FaceMaskDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations["file_name"].iloc[index])
        image = Image.open(img_path)

        tag = self.annotations["class"].iloc[index]
        tag = int(0 if tag == "No-Mask" else 1)
        tag = torch.tensor(tag)

        if self.transform:
            image = self.transform(image)

        return image, tag

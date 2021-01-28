import glob
import os
import shutil
import random

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

if __name__ == '__main__':

    # images_path = "/home/ambolt/Data/emily/faces/images"
    # image_names = glob.glob1(images_path, "*.jpg")
    #
    # n_images = len(image_names)
    # n_images_train = int(n_images * 0.8)
    # n_images_test = int(n_images * 0.2)
    #
    # images_names_for_test = random.shuffle(image_names)
    #
    # test_images = "/home/ambolt/Data/emily/faces/test_images"
    # train_images = "/home/ambolt/Data/emily/faces/train_images"

    label_filename = "/home/ambolt/Data/emily/faces/labels.csv"

    label_file_df = pd.read_csv(label_filename)

    print(label_file_df['class'].value_counts())
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = train_test_split(label_file_df, train_size=0.8, test_size=0.2)

    print("Train\n", train_df['class'].value_counts())
    print("Test\n", test_df['class'].value_counts())

    train_df.to_csv("/home/ambolt/Data/emily/faces/train.csv")
    test_df.to_csv("/home/ambolt/Data/emily/faces/test.csv")








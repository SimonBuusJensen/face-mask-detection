from utils.converters.mafa2kitti import mafa2kitti
from utils.converters.fddb2kitti import fddb2kitti
import os

"""
Author: Simon Buus Jensen
Original source and credit: https://github.com/NVIDIA-AI-IOT
"""


def main():
    """
    Script for converting the MAFA (Masked Faces) dataset and FDDB (Face Detection Data Set and Benchmark) Dataset.
    Download datasets at urls:
    MAFA: https://www.kaggle.com/ivandanilovich/medical-masks-dataset-images-tfrecords
    FDDB: http://vis-www.cs.umass.edu/fddb/

    The MAFA and FDDB datasets will be converted into KITTI format, where each images will be stored in a "images/"
    folder and each image has a corresponding label .txt file in a "labels/" folder.
    Labels will look something like the following:
    index:  0           1   2   3   4   5   6   7   8 9 10  11  12  13  14
    e.g.    "Masked"    _   _   _   587 173 614 200 _ _ _   _   _   _   _
    desc:   class name              x1  y1  x2  y2

    IMPORTANT:
    The MAFA and FDDB needs to be downloaded and structed like described in the following order
    for the script to work:
    See: https://github.com/NVIDIA-AI-IOT/face-mask-detection/blob/master/data_utils/data-tree.txt
    """
    # Point to where the MAFA and FDDB datasets are located
    mafa_base_dir = "/home/ambolt/Data/emily/MAFA Dataset"
    fddb_base_dir = "/home/ambolt/Data/emily/FDDB Dataset"

    # Specify where the converted images and labels will be stored
    output_dir = "/home/ambolt/Data/emily/faces_kitti Dataset/data_new"
    category_limit_mod = [3200, 3200]

    total_masks, total_no_masks = 0, 0
    # ----------------------------------------
    # MAFA Dataset Conversion
    # ----------------------------------------

    # The resize dims could potentially be optimized by finding a better aspect ratio
    mafa_resize_dims = (960, 544)

    annotation_file = os.path.join(mafa_base_dir, 'MAFA-Label-Train/LabelTrainAll.mat')
    mafa_base_dir = os.path.join(mafa_base_dir, 'train-images/images')
    kitti_label = mafa2kitti(annotation_file=annotation_file, mafa_base_dir=mafa_base_dir,
                             kitti_base_dir=output_dir, kitti_resize_dims=mafa_resize_dims,
                             category_limit=category_limit_mod)

    count_masks, count_no_masks = kitti_label.mat2data()
    total_masks += count_masks
    total_no_masks += count_no_masks

    # ----------------------------------------
    # FDDB Dataset Conversion
    # ----------------------------------------

    # The resize dims could potentially be optimized by finding a better aspect ratio
    fddb_resize_dims = (660, 544)

    annotation_path = os.path.join(fddb_base_dir, 'FDDB-folds')
    kitti_label = fddb2kitti(annotation_path=annotation_path, fddb_base_dir=fddb_base_dir,
                             kitti_base_dir=output_dir, kitti_resize_dims=fddb_resize_dims,
                             category_limit=category_limit_mod)
    count_masks, count_no_masks = kitti_label.fddb_data()
    total_masks += count_masks
    total_no_masks += count_no_masks

    print("----------------------------")
    print("Final: Total Mask Labelled:{}\nTotal No-Mask Labelled:{}".format(total_masks, total_no_masks))
    print("----------------------------")


if __name__ == '__main__':
    main()

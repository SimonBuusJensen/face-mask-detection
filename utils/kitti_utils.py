import glob
import os
import cv2
from utils.utility import create_csv_writer

mafa_path = "/home/ambolt/Data/emily/MAFA Dataset/kitti"


def open_label_file(sample_name, is_train):
    label_name = sample_name + ".txt"
    dir_name = "train" if is_train else "test"
    fp = os.path.join(mafa_path, dir_name, "labels", label_name)
    label_file = open(fp, "r")
    return label_file


def open_img_file(sample_name, is_train):
    img_name = sample_name + ".jpg"
    dir_name = "train" if is_train else "test"
    fp = os.path.join(mafa_path, dir_name, "images", img_name)
    img = cv2.imread(fp)
    return img


def get_sample(sample_name, is_train):
    img = open_img_file(sample_name, is_train)
    bbox = get_bounding_boxes(sample_name, is_train)
    class_name = get_class(sample_name, is_train)
    return img, bbox, class_name


def read_label(sample_name, is_train):
    label_file = open_label_file(sample_name, is_train)
    lines = label_file.readlines()
    # assert len(lines) == 1
    return lines[0]


def get_class(sample_name, is_train):
    line = read_label(sample_name, is_train)
    line_splits = line.split(" ")
    class_name = line_splits[0]
    return 1 if class_name == "Mask" else 0


def get_bounding_boxes(sample_name, is_train=True):
    line = read_label(sample_name, is_train)
    line_splits = line.split(" ")
    x1 = line_splits[4]
    y1 = line_splits[5]
    x2 = line_splits[6]
    y2 = line_splits[7]
    return map(int, [x1, y1, x2, y2])


def get_sample_names(train=True):
    dir_name = "train" if train else "test"
    img_names = glob.glob1(os.path.join(mafa_path, dir_name, "images"), "*.jpg")
    sample_names = [x.rstrip(".jpg") for x in img_names]
    return sample_names


def extract_face_images(out_dir=None, is_train=True):
    for sample_name in get_sample_names(is_train):
        # print(sample_name)
        img, bbox, class_name = get_sample(sample_name, is_train)
        x1, y1, x2, y2 = bbox
        face_img = img[y1:y2, x1:x2]
        try:
            cv2.imwrite(os.path.join(out_dir, sample_name + ".jpg"), face_img)
        except cv2.error as e:
            print(sample_name, e)


def get_csv_writer(out_dir):
    fname = "labels.csv"
    out_fp = os.path.join(out_dir, fname)
    columns = ["file_name", "class"]

    # If the csv-file exists we append to it
    if os.path.exists(out_fp):
        file = open(out_fp, "a+")
        file_writer = create_csv_writer(file, columns, sep=",", write_header=False)
    # If the csv-file does NOT exists we write to it and create a header row
    else:
        file = open(out_fp, "w")
        file_writer = create_csv_writer(file, columns, sep=",", write_header=True)
    return file_writer


def extract_labels(out_dir=None, is_train=True):
    file_writer = get_csv_writer(out_dir)

    for sample_name in get_sample_names(is_train):
        # print(sample_name)
        img, bbox, class_name = get_sample(sample_name, is_train)
        print(sample_name, class_name)
        out_dict = {
            "file_name": sample_name + ".jpg",
            "class": class_name
        }
        file_writer.writerow(out_dict)


if __name__ == '__main__':
    extract_labels("/home/ambolt/Data/emily/MAFA_faces/train/", is_train=True)

    # for sample_name in get_sample_names():
    #     print(sample_name)
    #     img, bbox, class_name = get_sample(sample_name, is_train=True)
    #     x1, y1, x2, y2 = bbox
    #     face_img = img[y1:y2, x1:x2]
    #     # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=2)
    #
    #     cv2.imshow(sample_name, face_img)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow(sample_name)

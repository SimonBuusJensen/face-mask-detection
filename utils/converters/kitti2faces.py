import glob
import os
import cv2
from utils.utility import create_csv_writer




def open_label_file(sample_name):
    label_name = sample_name + ".txt"
    fp = os.path.join(kitti_path, "data", "labels", label_name)
    label_file = open(fp, "r")
    return label_file


def open_img_file(sample_name):
    img_name = sample_name + ".jpg"
    fp = os.path.join(kitti_path, "data", "images", img_name)
    img = cv2.imread(fp)
    return img


def increase_bbox_size(bbox, px=20):
    x1, y1, x2, y2 = bbox
    half_px = int(px / 2)
    x1 = x1 - half_px
    y1 = y1 - half_px
    x2 = x2 + half_px
    y2 = y2 + half_px
    return (x1, y1, x2, y2)


def get_sample(sample_name, enlarge_bbox_by=0):
    img = open_img_file(sample_name)

    bbox = get_bounding_boxes(sample_name)

    if enlarge_bbox_by > 0:
        bbox = increase_bbox_size(bbox, enlarge_bbox_by)

    x1, y1, x2, y2 = bbox
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > img.shape[1]:
        x2 = img.shape[1]
    if y2 > img.shape[0]:
        y2 = img.shape[0]
    if x1 > x2:
        return None
    if y1 > y2:
        return None

    class_name = get_class(sample_name)
    return img, (x1, y1, x2, y2), class_name


def read_label(sample_name):
    label_file = open_label_file(sample_name)
    lines = label_file.readlines()
    # assert len(lines) == 1
    return lines[0]


def get_class(sample_name):
    line = read_label(sample_name)
    line_splits = line.split(" ")
    class_name = line_splits[0]
    return class_name


def get_bounding_boxes(sample_name):
    line = read_label(sample_name)
    line_splits = line.split(" ")
    x1 = line_splits[4]
    y1 = line_splits[5]
    x2 = line_splits[6]
    y2 = line_splits[7]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    return x1, y1, x2, y2


def get_sample_names():
    img_names = glob.glob1(os.path.join(kitti_path, "data", "images"), "*.jpg")
    sample_names = [x.rstrip(".jpg") for x in img_names]
    return sample_names


def extract_face_images(out_dir=None, enlarge_bbox_by=0):
    for sample_name in get_sample_names():
        # print(sample_name)
        sample = get_sample(sample_name, enlarge_bbox_by)
        if sample:
            img, bbox, class_name = sample
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


def extract_labels(out_dir=None):
    file_writer = get_csv_writer(out_dir)

    for sample_name in get_sample_names():

        sample = get_sample(sample_name)
        if sample:
            _, _, class_name = sample
            out_dict = {
                "file_name": sample_name + ".jpg",
                "class": class_name
            }
            file_writer.writerow(out_dict)


def main():

    kitti_data = "/home/ambolt/Data/emily/faces_kitti Dataset"
    output_dir = "/home/ambolt/Data/emily/faces"

    extract_labels()
    extract_face_images("/home/ambolt/Data/emily/faces/images_big", 50)


if __name__ == '__main__':
    main()

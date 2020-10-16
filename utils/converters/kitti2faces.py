import glob
import os
import cv2

from utils.utility import create_csv_writer, create_kitti_csv_writer


def main():
    # Path to the kitti formated data
    kitti_data = "/home/ambolt/Data/emily/faces_kitti Dataset/data"
    extra_px_around_face_pct = 25

    # Specify where the converted face images and labels should be stored
    output_dir = "/home/ambolt/Data/emily/faces2"

    try:
        os.makedirs(output_dir)
    except FileExistsError:
        print("Directory Already Exists")

    try:
        os.makedirs(os.path.join(output_dir, "images"))
    except FileExistsError:
        print("Directory Already Exists")

    # ----------------------------------------
    # Extract faces and labels from KITTI image Dataset
    # ----------------------------------------
    kitti2face(kitti_data_dir=kitti_data,
               out_dir=output_dir,
               extra_px_around_face_pct=extra_px_around_face_pct)


def kitti2face(kitti_data_dir, out_dir, extra_px_around_face_pct):

    annotation_files_dir = os.path.join(kitti_data_dir, 'labels')
    image_files_dir = os.path.join(kitti_data_dir, 'images')
    file_writer = create_kitti_csv_writer(out_dir)

    n_samples = len(get_sample_names(kitti_data_dir))
    for sample_i, sample_name in enumerate(get_sample_names(kitti_data_dir)):
        img_path = os.path.join(image_files_dir, sample_name + ".jpg")
        img = get_image(img_path)

        if img is None:
            continue

        label_path = os.path.join(annotation_files_dir, sample_name + ".txt")
        label = get_label(label_path)

        class_name, x1, y1, x2, y2 = label_to_line(label)
        bbox = (x1, y1, x2, y2)
        bbox = increase_bbox_size(bbox, extra_px_around_face_pct)
        bbox = check_bbox_dims(img, bbox)
        if bbox is None:
            continue

        face = extract_face(img, bbox)

        # Save the face image
        new_image_path = os.path.join(out_dir, "images", sample_name + ".jpg")
        cv2.imwrite(new_image_path, face)

        # Write to csv-file
        out_dict = {
            "file_name": sample_name + ".jpg",
            "class": class_name
        }
        file_writer.writerow(out_dict)

        if (sample_i + 1) % 100 == 0:
            print("Converted images and labels (" + str(sample_i + 1) + "/" + str(n_samples) + ")")


def get_sample_names(kitti_data_dir):
    img_names = glob.glob1(os.path.join(kitti_data_dir, "images"), "*.jpg")
    sample_names = [x.rstrip(".jpg") for x in img_names]
    return sample_names


def get_image(sample_img_path):
    fp = sample_img_path
    img = cv2.imread(fp)
    return img


def get_label(sample_label_path):
    fp = sample_label_path
    label_file = open(fp, "r")
    lines = label_file.readlines()
    return lines[0]


def label_to_line(label):
    line_splits = label.split(" ")

    class_name = line_splits[0]

    x1 = line_splits[4]
    y1 = line_splits[5]
    x2 = line_splits[6]
    y2 = line_splits[7]
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    return class_name, x1, y1, x2, y2


def increase_bbox_size(bbox, pct_increase=0):

    x1, y1, x2, y2 = bbox

    assert pct_increase >= 0 and pct_increase <= 100, \
        f"pct_increase should be greater than or equal 0 and lesser than or equal 100"

    width = abs(x2 - x1)
    height = abs(y2 - y1)

    if pct_increase > 0:
        pct_increase = (pct_increase + 100) / 100
        extra_width = int((width * pct_increase - width) / 2)
        extra_heigth = int((height * pct_increase - height) / 2)
        x1 = x1 - extra_width
        y1 = y1 - extra_heigth
        x2 = x2 + extra_width
        y2 = y2 + extra_heigth

    return (x1, y1, x2, y2)


def check_bbox_dims(img, bbox):
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

    return (x1, y1, x2, y2)


def extract_face(img, face_bbox):
    x1, y1, x2, y2 = face_bbox
    face_img = img[y1:y2, x1:x2]
    return face_img


if __name__ == '__main__':
    main()

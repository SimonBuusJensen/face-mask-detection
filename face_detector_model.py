import cv2
import random
import glob


def get_resize_scale(image, MAX_HEIGHT=512):
    img_height = image.shape[0]
    if img_height > MAX_HEIGHT:
        return MAX_HEIGHT / img_height
    else:
        return 1.


def resize(image):
    scale = get_resize_scale(image)
    image = cv2.resize(image, dsize=(int(image.shape[1] * scale), int(image.shape[0] * scale)))
    return image


def eyes_2_face(img, eyes):

    eye_1, eye_2 = eyes[:2]
    if eye_1[0] > eye_2[0]:
        tmp_eye = eye_2
        eye_2 = eye_1
        eye_1 = tmp_eye

    (x_eye_1, y_eye_1, w_eye_1, h_eye_1) = eye_1
    (x_eye_2, y_eye_2, w_eye_2, h_eye_2) = eye_2

    face_x1 = x_eye_1 - int(w_eye_1 * 2)
    face_y1 = y_eye_1 - int(h_eye_1 * 3)
    face_x2 = x_eye_2 + w_eye_2 + int(w_eye_1 * 2)
    face_y2 = y_eye_2 + int(h_eye_1 * 4)

    if face_x1 <= 0:
        face_x1 = 0
    if face_y1 <= 0:
        face_y1 = 0
    if face_x2 >= img.shape[1] - 1:
        face_x2 = img.shape[1] - 1
    if face_y2 >= img.shape[0] - 1:
        face_y2 = img.shape[0] - 1

    return (face_x1, face_y1, face_x2, face_y2)

def detect_face(img, face_cascade):

    # Resize the input image according to max allowed height
    img = resize(img)

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    eyes = face_cascade.detectMultiScale(gray, 1.1, 8)

    if len(eyes) == 2:
        face = eyes_2_face(gray, eyes)
        # Draw rectangle around the faces
        return face, eyes
    else:
        return None
        # cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 2)



# Load the cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Read a random input image
# images = glob.glob1("examples", "*.jpg")
# random.shuffle(images)
# img = cv2.imread('examples/' + images[0])
#
# face, eyes = detect_face(img, face_cascade)




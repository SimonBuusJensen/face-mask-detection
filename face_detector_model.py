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


# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read a random input image
images = glob.glob1("examples", "*.jpg")
random.shuffle(images)
img = cv2.imread('examples/' + images[0])

# Resize the input image according to max allowed height
img = resize(img)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
# cv2.imshow('img', img)
# cv2.waitKey()

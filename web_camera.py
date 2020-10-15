import numpy as np
import cv2
from face_detector_model import face_cascade, resize
from face_classifier_model import inference
import torch

if __name__ == "__main__":

    model = torch.load("/home/simon/projects/emily-face-mask-detection/models/google_LeNet_epoch_10.pth")
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        img = resize(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        (x, y, w, h) = faces[0]
        x1, y1, x2, y2 = x, y, x+w, y+h
        face_img = img[y1:y2, x1:x2]

        # Draw rectangle around the faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        pred = inference(face_img, model)
        # Display the resulting frame
        cv2.imshow('frame', img)
        cv2.imshow(pred, face_img)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
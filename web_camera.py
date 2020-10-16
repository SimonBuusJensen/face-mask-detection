import numpy as np
import cv2
from face_detector_model import face_cascade, detect_face
from face_classifier_model import inference
from face_classifier_pytorch_2 import Net
import torch
from PIL import Image

font = cv2.FONT_HERSHEY_SIMPLEX

class MaxSizeList(list):
    def __init__(self, maxlen):
        self._maxlen = maxlen

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)

if __name__ == "__main__":

    model = torch.load("/home/simon/projects/emily-face-mask-detection/models/custom5.pth",
                       map_location=torch.device('cpu'))
    cap = cv2.VideoCapture(0)
    preds = MaxSizeList(12)
    preds.append(0)
    preds.append(1)


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        # img = resize(frame)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray, 1.04, 10)

        detections = detect_face(frame, face_cascade)

        if detections is not None:
            face, eyes = detections
            (x, y, x2, y2) = face
            face_img = frame[y:y2, x:x2].copy()
            cv2.imshow('face', face_img)
            pred = inference(Image.fromarray(face_img), model)
            preds.append(pred)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

            for eye in eyes:
                (x, y, w, h) = eye
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if np.mean(preds) <= 0.5:
            text = "Mask"
            text_color = (0, 255, 0)
        else:
            text = "No Mask"
            text_color = (0, 0, 255)
        cv2.putText(frame, text, (20, 50), font, 1, text_color, 2, cv2.LINE_AA)
            # cv2.imshow("face", face_img)

            # Draw rectangle around the faces

        # Display the resulting frame
        print(preds)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
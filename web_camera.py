import numpy as np
import cv2
from FaceDetector import FaceDetector
# from face_classifier_pytorch_2 import Net
from face_classifier_model import inference, Net
import torch
from PIL import Image

font = cv2.FONT_HERSHEY_SIMPLEX

class MaxSizeList(list):

    def __init__(self, maxlen):
        self._maxlen = maxlen
        self.append(0)

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)

if __name__ == "__main__":

    model = torch.load("/home/simon/Ambolt/emily/emily-face-mask-detection/models/custom5.pth",
                       map_location=torch.device('cpu'))
    face_detector = FaceDetector()
    preds = MaxSizeList(12)

    cap = cv2.VideoCapture(0)
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        detections = face_detector.detect_face(frame)

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
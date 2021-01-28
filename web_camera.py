import cv2

import numpy as np
from PIL import Image

from FaceDetector import FaceDetector
from model import ResNetModel, CustomModel
from predict import inference
from transforms import Transformer

font = cv2.FONT_HERSHEY_SIMPLEX


class MaxSizeList(list):

    def __init__(self, maxlen):
        self._maxlen = maxlen
        self.append(0)

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)


if __name__ == "__main__":

    face_detector = FaceDetector()
    face_classifier_model = ResNetModel()
    face_classifier_model.load_model('models/2021-01-28/resnet_model_epoch_5_acc_97.37.pth', 'cpu')
    transformer = Transformer()

    preds = MaxSizeList(12)
    cap = cv2.VideoCapture(0)
    while (True):

        # Capture frame-by-frame
        ret, frame = cap.read()
        detections = face_detector.detect_face(frame)

        if detections is not None:
            face, eyes = detections
            (x, y, x2, y2) = face
            face_img = frame[y:y2, x:x2].copy()
            cv2.imshow('face', face_img)
            pred_class, pred_conf = inference(face_classifier_model, Image.fromarray(face_img), transformer.test_transforms())
            preds.append(pred_conf)
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)

            for eye in eyes:
                (x, y, w, h) = eye
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        print(np.mean(preds))
        if np.mean(preds) >= 0.90:
            text = "No Mask"
            text_color = (0, 255, 0)
        else:
            text = "Mask"
            text_color = (0, 0, 255)

        cv2.putText(frame, text, (20, 50), font, 1, text_color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

import cv2

import numpy as np
from PIL import Image

from ml.model import Model
from ml.transforms import test_transforms
from ml.predictor import Predictor


font = cv2.FONT_HERSHEY_SIMPLEX


class MaxSizeList(list):

    def __init__(self, maxlen):
        self._maxlen = maxlen
        self.append(0)

    def append(self, element):
        self.__delitem__(slice(0, len(self) == self._maxlen))
        super(MaxSizeList, self).append(element)


if __name__ == "__main__":

    model = Model()
    model.load_model("models/2021-03-11-3/resnet_18_epoch_153_acc_93.53.pth")
    predictor = Predictor(model)
    preds = MaxSizeList(10)

    cap = cv2.VideoCapture(0)
    while (True):

        ret, frame = cap.read()
        img = Image.fromarray(frame)
        pred = predictor.predict(img)

        preds.append(pred)
        mean_pred = round(float(np.mean(preds)), 2)
        if mean_pred < 50:
            text = "No Mask"
            text_color = (0, 0, 255)
        else:
            text = "Mask"
            text_color = (0, 255, 0)

        cv2.putText(frame, text, (20, 50), font, 1, text_color, 2, cv2.LINE_AA)
        cv2.putText(frame, str(mean_pred), (20, 100), font, 1, text_color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

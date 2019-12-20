import cv2
import os
from pathlib import Path

def read_images(path, labels, show_im=False):
    for label in labels:
        dir = os.path.join(path, label)
        for img in os.listdir(dir):
            img_path = os.path.join(dir, img)
            im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if(show_im == True):
                cv2.imshow("image", im)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

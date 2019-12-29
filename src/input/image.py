import cv2
import os
from pathlib import Path

def read_images(path, labels, show_im=False, flag=cv2.IMREAD_GRAYSCALE):
    for label in labels:
        dir = os.path.join(path, label)
        for img in os.listdir(path=dir):
            img_path = os.path.join(dir, img)
            im = cv2.imread(filename=img_path, flags=flag)
            if(show_im == True):
                try:
                    cv2.imshow(winname="image", mat=im)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error:
                    print("cv2.error: Skipping image {}".format(img))
                    pass
    im = cv2.resize(im, (100, 100))
    return im

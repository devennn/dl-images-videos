import cv2
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import pickle
import math
import sys
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

def load_new_images():
    im_size = 100
    new_im_array = []

    path = Path('.').parent.absolute()
    dog_path = os.path.join(path, 'datasets', 'dog_test.jpg')
    cat_path = os.path.join(path, 'datasets', 'cat_test.jpg')

    img_dog = img_to_array(load_img(dog_path, target_size=(im_size, im_size)))
    img_dog = (img_dog.reshape(-1, im_size, im_size, 1).astype('float32'))
    img_dog = img_dog  / 255
    new_im_array.append(img_dog)
    img_cat = img_to_array(load_img(cat_path, target_size=(im_size, im_size)))
    img_cat = (img_cat.reshape(-1, im_size, im_size, 1).astype('float32'))
    img_cat = img_cat  / 255
    new_im_array.append(img_cat)

    return new_im_array

def load_petimage():
    print("=== Start Loading PetImages ===")
    labels = ['Cat', 'Dog']
    path = Path('.').parent.absolute()
    path = os.path.join(path, 'datasets', 'PetImages')
    print("=== Loading from : {} ===".format(path))

    return read_images(path, labels)

def read_images(path, labels, show_im=False, im_size = 100):
    training_data = []
    for label in labels:
        data_dir = os.path.join(path, label)
        print("Loading: {}".format(data_dir))
        # get the classification  (0 or a 1). 0=dog 1=cat
        class_num = labels.index(label)

        for img in tqdm(os.listdir(data_dir)):
            try:
                img_path = os.path.join(path, label, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (im_size, im_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

    return split_features_labels(training_data, im_size)

def split_features_labels(training_data, im_size):
    random.shuffle(training_data)
    X, y = [], []
    for features, label in training_data:
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, im_size, im_size, 1)
    y = np.array(y)

    return format_data(X, y)

def format_data(X, y):
    X = X.astype('float32')
    X = X / 255
    train_len = math.ceil(0.7 * len(X))
    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len + 1:], y[train_len + 1:]

    return X_train, X_test, y_train, y_test

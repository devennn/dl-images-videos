from tensorflow.keras.datasets import cifar100
from tensorflow import keras
from tensorflow.keras import backend as K
from . import image
from pathlib import Path
import os
import numpy as np
from six.moves import cPickle
import sys

# SSL license
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_example(sample_name):
    print("=== Loading Example ===")
    if(sample_name == "cifar10"):
        return __load_cifar10_or_cifar100(sample_name)
    elif(sample_name == "cifar100"):
        return __load_cifar10_or_cifar100(sample_name)
    elif(sample_name == "PetImages"):
        return load_petimage(sample_name)

def __load_cifar10_or_cifar100(sample_name):
    print("=== Start Loading {} ===".format(sample_name))
    # Load the entire data set
    path = Path('.').parent.absolute()
    if(sample_name == "cifar10"):
        path = os.path.join(path, 'datasets\cifar-10-batches-py')
        (X_train, y_train), (X_test, y_test) = load_cifar10_data(path)
    elif(sample_name == "cifar100"):
        path = os.path.join(path, 'datasets\cifar-100-python')
        (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # Normalize data set to 0-to-1 range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y=y_train, num_classes=10)
    y_test = keras.utils.to_categorical(y=y_test, num_classes=10)

    return X_train, X_test, y_train, y_test

def load_petimage(sample_name):
    print("=== Start Loading PetImages ===")
    dataset_folder = 'datasets'
    labels = ['Cat', 'Dog']

    # processing directory
    path = Path(dataset_folder).parent.absolute()
    path = os.path.join(path, dataset_folder, sample_name)
    print("=== Loading from : {} ===".format(path))

    return image.read_images(path, labels, show_im=True)

def load_cifar10_data(dir):
    num_train_samples = 50000
    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(dir, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath=fpath)

    fpath = os.path.join(dir, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def load_batch(fpath, label_key='labels'):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

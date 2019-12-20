from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow import keras
from . import image_prep as prep
from pathlib import Path
import os

# SSL license
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class Sample_Input:

    def __init__(self, sample_name):
        self.sample_name = sample_name

    def load_example(self):
        print("=== Loading Example ===")
        if(self.sample_name == "cifar10"):
            return self.__load_cifar10_or_cifar100(name=self.sample_name)
        elif(self.sample_name == "cifar100"):
            return self.__load_cifar10_or_cifar100(name=self.sample_name)
        elif(self.sample_name == "PetImages"):
            return self.__load_petimage()

    def __load_cifar10_or_cifar100(self, name):
        print("=== Start Loading {} ===".format(name))
        # Load the entire data set
        if(name == "cifar10"):
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        elif(name == "cifar100"):
            (X_train, y_train), (X_test, y_test) = cifar100.load_data()

        # Normalize data set to 0-to-1 range
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return X_train, X_test, y_train, y_test

    def __load_petimage(self):
        print("=== Start Loading PetImages ===")
        dataset_folder = 'datasets'
        labels = ['Cat', 'Dog']

        # processing directory
        path = Path(dataset_folder).parent.absolute()
        path = os.path.join(path, dataset_folder)
        path = os.path.join(path, self.sample_name)
        print("=== Loading from : {} ===".format(path))

        return prep.read_images(path, labels, show_im=True)

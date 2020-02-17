from keras.datasets import cifar10
from keras.datasets import cifar100
import tensorflow as tf
import sys

def load_cifar10():
    # Load the entire data set
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize data set to 0-to-1 range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    print(X_train.shape)
    sys.exit()

    # Convert class vectors to binary class matrices
    class_num = 10
    y_train = tf.keras.utils.to_categorical(y_train, class_num)
    y_test = tf.keras.utils.to_categorical(y_test, class_num)

    return X_train, y_train, X_test, y_test, class_num

def load_cifar100():
    # Load the entire data set
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # Normalize data set to 0-to-1 range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    class_num = 100
    y_train = tf.keras.utils.to_categorical(y_train, class_num)
    y_test = tf.keras.utils.to_categorical(y_test, class_num)

    return X_train, y_train, X_test, y_test, class_num

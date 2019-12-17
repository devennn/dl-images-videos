from tensorflow.keras.datasets import cifar10
from tensorflow import keras

# SSL license
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_sample_input():
    # Load the entire data set
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize data set to 0-to-1 range
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test

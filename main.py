from input import datasets
from model import define_sequential
from model import train
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import model_from_json
import sys
import numpy as np
from pathlib import Path
import os

def in_out_shape(X_train, y_train):
    try:
        data_shape = X_train[1].shape
    except IndexError:
        data_shape = X_train.shape

    print('Data Shape: {}'.format(data_shape))

    out_class = []
    for i in y_train:
        if i not in out_class:
            out_class.append(i)
    data_class = len(out_class)
    return data_shape, data_class

def load_model():
    # Load the json file that contains the model's structure
    path = Path('.').parent.absolute()
    model_file = os.path.join(path, 'results', 'model_structure_local.json')
    weights_file = os.path.join(path, 'results', 'model_weights_local.h5')

    with open(model_file, 'r') as f:
        model_structure = f.read()
    # Recreate the Keras model object from the json data
    model = model_from_json(model_structure)
    # Re-load the model's trained weights
    model.load_weights(weights_file)
    return model

def predict_class(model, class_labels):
    new_im_array = datasets.load_new_images()
    f = open('predictions.txt', 'w')
    for img in new_im_array:
        results = model.predict(img)
        single_result = results[0]
        most_likely_class_index = int(np.argmax(single_result))
        class_likelihood = single_result[most_likely_class_index]
        class_label = class_labels[most_likely_class_index]

        str = 'This is image is a {} - Likelihood: {:2f}\n'.format(
                    class_label, class_likelihood
                )
        f.write(str)
        print(str)

    f.close()

if __name__ == '__main__':

    # X_train, X_test, y_train, y_test = datasets.load_petimage()
    #
    # data_shape, data_class = in_out_shape(X_train, y_train)
    # print('=== Input Shape: {} Output Class: {} ==='.format(data_shape, data_class))
    # model = define_sequential(input_shape=data_shape, output_class=data_class,
    #         hidden_neurons=[32, 64], dropout_val=0.3, in_activation='relu',
    #         lyr_activation='relu', out_activation='softmax',
    #         loss='sparse_categorical_crossentropy', optimizer='adam',
    #         show_summary=True
    #     )
    #
    # model = train(model, tr_dat=(X_train, y_train), tst_dat=(X_test, y_test),
    #     batch_size=64, epochs=1
    # )

    class_labels = ['cat', 'dog']
    model = load_model()
    predict_class(model, class_labels)

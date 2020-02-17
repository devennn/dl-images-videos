from pathlib import Path
import os
import sys
from model import define_model
from load import *
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from logs.set_logger import set_logger
logger = set_logger(name=__name__, level='Info')

if __name__ == '__main__':
    logger.info('Loading Dataset')
    X_train, y_train, X_test, y_test, class_num = load_cifar100()
    logger.info('Defining Model')
    model = define_model(class_num=class_num)

    logger.info('Saving structure')
    path = Path('.').parent.absolute()
    model_structure = model.to_json()
    structure_path = os.path.join(path, 'results', 'model_structure.json')
    with open(structure_path, 'w') as f:
        f.write(model_structure)

    logger.info('Start Fitting')
    model.fit(
        X_train, y_train,
        batch_size=64, epochs=50,
        validation_data=(X_test, y_test),
        shuffle=True
    )

    logger.info('Saving weights')
    # Save neural network structure and weights
    weight_path = os.path.join(path, 'results', 'model_weights.h5')
    model.save_weights(weight_path)

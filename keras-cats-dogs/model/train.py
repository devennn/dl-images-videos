from pathlib import Path

# Disable Warnings
import os
os.environ['KMP_WARNINGS'] = 'off'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable tensorflow deprecation warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def train(model, tr_dat, tst_dat, batch_size=32, epochs=30):

    model.fit(
        tr_dat[0],
        tr_dat[1],
        batch_size=batch_size,
        epochs=epochs,
        validation_data=tst_dat,
        shuffle=True
    )

    save_structure_weight(model)
    return model

def save_structure_weight(model):
    print("Saving Model...")
    model_structure = model.to_json()
    with open('model_structure.json', 'w') as f:
        f.write(model_structure)

    print("Saving Weights...")
    model.save_weights('model_weights.h5')

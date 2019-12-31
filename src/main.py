from input import datasets
from intelligent.define_model import Setup_Sequential
from intelligent.train import Train_Model
from utils import load_parameters
import sys

model_par, fit_par = load_parameters.load_parameters()

X_train, X_test, y_train, y_test = datasets.load_example(
                                        sample_name=sys.argv[1]
                                    )

sys.exit("Stop at data processing...")

seq = Setup_Sequential()
seq.input_shape = (100, 100, 1)
model = seq.define_model(nodes=[32, 64])

tr = Train_Model(epochs=int(sys.argv[2]))
model = tr.train(
    model,
    tr_dat=(X_train, y_train),
    tst_dat=(X_test, y_test)
)

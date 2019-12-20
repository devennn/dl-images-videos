from input.sample import Sample_Input
from intelligent.setup_model import Setup_Sequential
from intelligent.train_model import Train_Model
import sys

smp = Sample_Input(sample_name=sys.argv[1])
X_train, X_test, y_train, y_test = smp.load_example()

seq = Setup_Sequential()
model = seq.define_model(nodes=[32, 64])

tr = Train_Model(epochs=sys.argv[2])
model = tr.train(model, tr_dat=(X_train, y_train),
    tst_dat=(X_test, y_test)
)

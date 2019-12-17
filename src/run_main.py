from input.process_input import load_sample_input
from intelligent.setup_model import Setup_Sequential
from intelligent.train_model import Train_Model

X_train, X_test, y_train, y_test = load_sample_input()

seq = Setup_Sequential()
model = seq.define_model(nodes=[32, 64])

tr = Train_Model(epochs=1)
model = tr.train(model, tr_dat=(X_train, y_train),
    tst_dat=(X_test, y_test)
)

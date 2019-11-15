from model import vanilla, stacked, bidirectional, cnn, convlstm
from preparation import split_sequence
from numpy import array

raw_seq = [10,20,30,40,50,60,70,80,90]
# for vanilla, stacked, bidirectional

n_steps = 3
X, y = split_sequence(raw_seq, n_steps)
n_features = 1 # univariate = one variable.
X = X.reshape((X.shape[0], X.shape[1], n_features))

models = [vanilla, stacked, bidirectional]

for m in models:
    model = m(X, y, n_steps, n_features)

    x_input = array([70,80,90])
    x_input = x_input.reshape((1,n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(m.__name__, yhat)

# for cnn model
n_steps = 4
X, y = split_sequence(raw_seq, n_steps)
n_features = 1
n_seq = 2
n_seq_steps = 2

models2 = [cnn,]

for m in models2:
    model = m(X, y, n_seq, n_seq_steps, n_features)
    x_input = array([60, 70, 80, 90])
    x_input = x_input.reshape((1, n_seq, n_seq_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(m.__name__, yhat)

# for convlstm
models3 = [convlstm,]

for m in models3:
    model = m(X, y, n_seq, 1, n_seq_steps, n_features)
    x_input = array([60, 70, 80, 90])
    x_input = x_input.reshape((1, n_seq, 1, n_seq_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(m.__name__, yhat)

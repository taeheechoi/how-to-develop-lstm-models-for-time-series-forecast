from model import vanilla, stacked, bidirectional
from preparation import split_sequences_multi_input_series, split_sequences_multi_parallel_series
from numpy import array, hstack

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))

dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps = 3
X, y = split_sequences_multi_input_series(dataset, n_steps)
n_features = X.shape[2]

models = [vanilla, stacked, bidirectional]

for m in models:
    model = m(X, y, n_steps, n_features)

    x_input = array([[80, 85], [90, 95], [100, 105]])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(m.__name__, yhat)

X, y = split_sequences_multi_parallel_series(dataset, n_steps)
n_features = X.shape[2]

for m in models:
    model = m(X, y, n_steps, n_features, n_output=3)

    x_input = array([[70,75,145], [80, 85,165], [90,95,185]])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(m.__name__, yhat)
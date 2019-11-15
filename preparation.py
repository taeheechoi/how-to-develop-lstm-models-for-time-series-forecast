from numpy import array, hstack

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_idx = i + n_steps
        # print(end_idx, len(sequence)-1)
        # 3 8
        # 4 8
        # 5 8
        # 6 8
        # 7 8
        # 8 8
        # 9 8
        if end_idx > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

# raw_seq = [10,20,30,40,50,60,70,80,90]
# n_steps = 3
# X, y = split_sequence(raw_seq, n_steps)
# print(X, y)
# [[10 20 30]
#  [20 30 40]
#  [30 40 50]
#  [40 50 60]
#  [50 60 70]
#  [60 70 80]] [40 50 60 70 80 90]


def split_sequences_multi_input_series(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
    # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :-1] # [row, column] ':-1' means except last, '-1:' means last only
        seq_y = sequences[end_ix-1, -1] # [row, column] '-1' means last only

        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y) # rows for time steps and columns for parallel series

# in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
# in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
# out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
# in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# out_seq = out_seq.reshape((len(out_seq), 1))
# dataset = hstack((in_seq1, in_seq2, out_seq))

# n_steps = 3
# X, y = split_sequences_multi_input_series(dataset, n_steps)
# print(X, y)
# [[[10 15]
#   [20 25]
#   [30 35]]

#  [[20 25]
#   [30 35]
#   [40 45]]

#  [[30 35]
#   [40 45]
#   [50 55]]

#  [[40 45]
#   [50 55]
#   [60 65]]

#  [[50 55]
#   [60 65]
#   [70 75]]

#  [[60 65]
#   [70 75]
#   [80 85]]

#  [[70 75]
#   [80 85]
#   [90 95]]] [ 65  85 105 125 145 165 185]


def split_sequences_multi_parallel_series(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        end_idx = i + n_steps
        if end_idx > len(sequences) -1:
            break
        seq_x, seq_y = sequences[i:end_idx, :], sequences[end_idx, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

in_seq1 = array([10, 20, 30, 40, 50, 60, 70, 80, 90])
in_seq2 = array([15, 25, 35, 45, 55, 65, 75, 85, 95])
out_seq = array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
out_seq = out_seq.reshape((len(out_seq), 1))
dataset = hstack((in_seq1, in_seq2, out_seq))

n_steps = 3
X, y = split_sequences_multi_parallel_series(dataset, n_steps)
# for i in range(len(X)):
#     print(X[i], y[i])
# [[10 15 25]
#  [20 25 45]
#  [30 35 65]] [40 45 85]
# [[20 25 45]
#  [30 35 65]
#  [40 45 85]] [ 50  55 105]
# [[ 30  35  65]
#  [ 40  45  85]
#  [ 50  55 105]] [ 60  65 125]
# [[ 40  45  85]
#  [ 50  55 105]
#  [ 60  65 125]] [ 70  75 145]
# [[ 50  55 105]
#  [ 60  65 125]
#  [ 70  75 145]] [ 80  85 165]
# [[ 60  65 125]
#  [ 70  75 145]
#  [ 80  85 165]] [ 90  95 185]
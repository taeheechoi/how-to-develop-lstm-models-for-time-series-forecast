from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import array, hstack

from preparation import split_sequences_multi_input_series

in_seq1 = array([10,20,30,40,50,60,70,80,90])
in_seq2 = array([15,25,35,45,55,65,75,85,95])
out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])


in_seq1 = in_seq1.reshape((len(in_seq1), 1))
# print(in_seq1)
# [[10]
#  [20]
#  [30]
#  [40]
#  [50]
#  [60]
#  [70]
#  [80]
#  [90]]
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
# print(in_seq2)
# [[15]
#  [25]
#  [35]
#  [45]
#  [55]
#  [65]
#  [75]
#  [85]
#  [95]]
out_seq = out_seq.reshape((len(out_seq), 1))
# print(out_seq)
# [[ 25]
#  [ 45]
#  [ 65]
#  [ 85]
#  [105]
#  [125]
#  [145]
#  [165]
#  [185]]
dataset = hstack((in_seq1, in_seq2, out_seq))
# print(dataset)
# [[ 10  15  25]
#  [ 20  25  45]
#  [ 30  35  65]
#  [ 40  45  85]
#  [ 50  55 105]
#  [ 60  65 125]
#  [ 70  75 145]
#  [ 80  85 165]
#  [ 90  95 185]]

n_steps = 3
X, y = split_sequences_multi_input_series(dataset, n_steps)
print(X.shape, y.shape) # (7, 3, 2) 7: number of samples, 3: number of time steps per sample, 2: number of parallel time series or variables
n_features = X.shape[2]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

x_input = array([[80, 85], [90,95], [100,105]])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

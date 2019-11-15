# convolutional neural network
# CNN is used to interpret subsequences of input that together are provided as a sequence to an LSTM model to interpret

from keras.layers import LSTM, Bidirectional, Dense, Flatten, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from numpy import array

from preparation import split_sequence

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 4
X, y = split_sequence(raw_seq, n_steps)
n_features = 1
n_seq = 2
n_seq_steps = 2
X = X.reshape((X.shape[0], n_seq, n_seq_steps, n_features))

model= Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_seq_steps, n_features)))
# filters: number of reads or interpretations of the input sequence
# kernel size: number of time steps included of each read operation of the input sequence
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# max pooling layer that distills the filter maps down to 1/2 of their size that includes the most salient features
model.add(TimeDistributed(Flatten()))
# flattened down to a single one-dimensional vector to be used as a single input time step to the LSTM
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, n_seq_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
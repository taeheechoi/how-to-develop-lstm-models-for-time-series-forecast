# convolutional reading of input is built directly into each LSTM unit, developed for reading two-dimensional spatial-temporal data
# input as a sequence of two-dimensional images

from keras.layers import LSTM, Bidirectional, ConvLSTM2D, Dense, Flatten
from keras.models import Sequential
from numpy import array

from preparation import split_sequence

raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 4
X, y = split_sequence(raw_seq, n_steps)
n_features = 1
n_seq = 2
n_seq_steps = 2
X = X.reshape((X.shape[0], n_seq, 1, n_seq_steps, n_features)) # [samples, timesteps] --> [samples, timesteps, rows, columns, features]
# columns: the number of time steps for each subsequence, or n_seq_steps
# row: 1 as one dimensional data

model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_seq_steps, n_features)))
model.add(Flatten())
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

x_input = array([60, 70, 80, 90])
x_input = x_input.reshape((1, n_seq, 1, n_seq_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)

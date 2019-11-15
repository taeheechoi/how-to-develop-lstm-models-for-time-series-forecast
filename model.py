# Reference: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
from keras.layers import (LSTM, Bidirectional, ConvLSTM2D, Dense, Flatten,
                          TimeDistributed)
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Sequential
from numpy import array


def vanilla(X, y, n_steps, n_features, n_output=1): # [samples, timesteps] --> [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

def stacked(X, y, n_steps, n_features, n_output=1): # [samples, timesteps] --> [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

def bidirectional(X, y, n_steps, n_features, n_output=1): # [samples, timesteps] --> [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu', input_shape=(n_steps, n_features))))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

def cnn(X, y, n_seq, n_seq_steps, n_features, n_output=1): # [samples, timesteps] --> [samples, subsequences, timesteps, features]
    X = X.reshape((X.shape[0], n_seq, n_seq_steps, n_features))
    model= Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_seq_steps, n_features)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

def convlstm(X, y, n_seq, rows, n_seq_steps, n_features, n_output=1): # [samples, timesteps] --> [samples, timesteps, rows, columns, features]
    X = X.reshape((X.shape[0], n_seq, rows, n_seq_steps, n_features))
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, rows, n_seq_steps, n_features)))
    model.add(Flatten())
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=200, verbose=0)
    return model

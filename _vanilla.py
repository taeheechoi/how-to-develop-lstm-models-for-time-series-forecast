# LSTM model that has a single hidden layer of LSTM units

from keras.layers import LSTM, Bidirectional, Dense
from keras.models import Sequential
from numpy import array

from preparation import split_sequence

# print(X)
# [[10 20 30]
#  [20 30 40]
#  [30 40 50]
#  [40 50 60]
#  [50 60 70]
#  [60 70 80]]

# print(X.shape[0], X.shape[1]) # Row Column 6 3
raw_seq = [10,20,30,40,50,60,70,80,90]
n_steps = 3
X, y = split_sequence(raw_seq, n_steps)
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# print(X)
# [[[10]
#   [20]
#   [30]]

#  [[20]
#   [30]
#   [40]]
# ...
#  [[60]
#   [70]
#   [80]]]

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features))) # the number of time steps and the number of features.
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse') #adam version of stochastic gradient descent, mean squared error
model.fit(X, y, epochs=200, verbose=0) # verbose=1 6/6 [==============================] - 0s 500us/step - loss: 0.5739
x_input = array([70,80,90])
x_input = x_input.reshape((1,n_steps, n_features))
# print(x_input)
# [[[70]
#   [80]
#   [90]]]
yhat = model.predict(x_input, verbose=0)
print(yhat)

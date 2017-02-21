from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers.core import Lambda
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.visualize_util import plot

"""Model Visualization"""
row, col, ch = (45, 160, 3)

model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(row, col, ch)))

model.add(Convolution2D(24, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(300, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(20, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(1))
adam = Adam(lr=1e-4)
model.compile(loss='mse', optimizer=adam)

plot(model, to_file='model.png')

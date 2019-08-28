import numpy as np
from ArtNet.layers import Input, Dense
from ArtNet.lib import r_squared
from ArtNet.losses import MeanSquaredError
from ArtNet.models import Sequential
from ArtNet.optimizers import Adam
from keras.datasets import boston_housing
from matplotlib import pyplot

# Data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Model
model = Sequential()
model.add(Input(input_shape=13))
model.add(Dense(nodes=26, activation="ReLU"))
# model.add(BatchNormalization())
# model.add(Activation("Tanh"))
# model.add(Dense(nodes=13, activation="ReLU"))
# model.add(BatchNormalization())
# model.add(Activation("Tanh"))
# model.add(Dropout())
model.add(Dense(nodes=1))
# GradientDescent(beta=0.9)
model.compile(loss=MeanSquaredError(), optimizer=Adam())
model.fit(x_train, y_train, epochs=5, batch_size=4)

# Evaluate
y_pred = model.predict(x_test, batch_size=x_test.shape[0], verbose=0)
print(y_pred)
print(y_test.reshape(1, -1))
print(np.round(r_squared(y_test, y_pred) * 100, 2), "%")
pyplot.plot(y_pred, color="blue")
pyplot.plot(y_test, color="green")
pyplot.show()

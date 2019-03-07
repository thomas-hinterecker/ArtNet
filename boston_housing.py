from NeuralNetwork.lib import r_squared
import numpy as np
from NeuralNetwork.layers import Input, Dense
from NeuralNetwork.lib import r_squared
from NeuralNetwork.losses import MeanSquaredError
from NeuralNetwork.models import Sequential
from NeuralNetwork.optimizers import Adam
# Data
from keras.datasets import boston_housing
from matplotlib import pyplot

# np.seterr(all='ignore')

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# x_train = normalize(x_train, axis=0)
# y_train = normalize(y_train.reshape(-1, 1))

# x_test = normalize(x_test, axis=0)
# y_test = normalize(y_test.reshape(-1, 1))


# print(x_train.shape)
# print(y_train.shape)

# Neural Network
# sys.path.append("NeuralNetwork")

model = Sequential()
model.add(Input(features=13))
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

y_pred = model.predict(x_test, batch_size=x_test.shape[0], verbose=0)
print(y_pred)
print(y_test.reshape(1, -1))
print(np.round(r_squared(y_test, y_pred) * 100, 2), "%")
pyplot.plot(y_pred, color="blue")
pyplot.plot(y_test, color="green")
pyplot.show()

import sys
import numpy as np
from matplotlib import pyplot
#np.seterr(all='ignore')

## Data
from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

from sklearn.preprocessing import normalize
#x_train = normalize(x_train, axis=0)
#y_train = normalize(y_train.reshape(-1, 1))

#x_test = normalize(x_test, axis=0)
#y_test = normalize(y_test.reshape(-1, 1))


#print(x_train.shape)
#print(y_train.shape)

## Neural Network
#sys.path.append("NeuralNetwork")
from NeuralNetwork.models import Sequential
from NeuralNetwork.layers import Input, Dense, Activation, BatchNormalization, Dropout
from NeuralNetwork.activations import ReLU, Sigmoid, Softmax, Tanh, Leaky_ReLU, Linear
from NeuralNetwork.regularizers import L2
from NeuralNetwork.losses import MeanSquaredError, BinaryCrossEntropy
from NeuralNetwork.optimizers import GradientDescent, RMSprop, Adam
from NeuralNetwork.lib import r_squared

model = Sequential()
model.add(Input(features=13))
model.add(Dense(nodes=26, activation="ReLU"))
#model.add(BatchNormalization())
#model.add(Activation("Tanh"))
#model.add(Dense(nodes=13, activation="ReLU"))
#model.add(BatchNormalization())
#model.add(Activation("Tanh"))
#model.add(Dropout())
model.add(Dense(nodes=1))
model.compile(loss=MeanSquaredError(), optimizer=Adam()) #GradientDescent(beta=0.9)
model.fit(x_train, y_train, epochs=5, batch_size=4)

y_pred = model.predict(x_test, batch_size=x_test.shape[0], verbose=0)
print(y_pred)
print(y_test.reshape(1,-1))
print(np.round(r_squared(y_test, y_pred) * 100, 2) , "%")
pyplot.plot(y_pred, color="blue")
pyplot.plot(y_test, color="green")
pyplot.show()

quit()

# ## Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

model = Sequential()
model.add(Dense(20, input_dim=13, activation='relu'))
model.add(Dropout())
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="adam")
model.fit(x_train, y_train, epochs=3, batch_size=5)

y_pred = model.predict(x_test, batch_size=x_test.shape[0], verbose=0)
print(np.round(r_squared(y_test, y_pred) * 100, 2) , "%")
pyplot.plot(y_pred, color="blue")
pyplot.plot(y_test, color="green")
pyplot.show()
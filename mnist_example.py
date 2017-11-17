from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
#np.seterr(all='ignore')

## Data
mnist = fetch_mldata('MNIST original')
classes = np.array(range(0, 10))
lb = preprocessing.LabelBinarizer()
lb.fit(classes)
mnist.target = lb.transform(mnist.target)
X_train, X_test, y_train, y_test = train_test_split(mnist.data/255, mnist.target, test_size=0.2)

#print(X_train.shape)
#print(y_train.shape)


## Neural Network
from ArtNet.models import Sequential
from ArtNet.layers import Input, Dense, Activation, Dropout, BatchNormalization
from ArtNet.activations import ReLU, Sigmoid, Softmax, Tanh, Leaky_ReLU, Linear
from ArtNet.regularizers import L2
from ArtNet.losses import MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from ArtNet.optimizers import GradientDescent, RMSprop, Adam
from ArtNet.lib import r_squared

model = Sequential()
model.add(Input(features=784))
model.add(Dense(nodes=100, activation="Linear")) #, kernel_regularizer=L2()
model.add(Activation("ReLU"))
#model.add(BatchNormalization())
model.add(Dense(nodes=10, activation=Softmax()))
model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(), metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=500, validation_data=(X_test, y_test))

# print('Weights Layer 1: ', model.W[1])
# print('Biases Layer 1: ', model.b[1])
# print('Weights Layer 2: ', model.W[2])
# print('Biases Layer 2: ', model.b[2])

y_pred = model.predict(X_test, batch_size=500, verbose=0)

#print("Accuracy:", (1 - np.sum(np.abs(y_test - np.round(np.squeeze(y_pred)))) / y_test.shape[0]) * 100, '%')
print("Accuracy:", np.round((1 - np.sum(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1)) / X_test.shape[0]) * 100, 2), "%")

# ## Keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, BatchNormalization
# #from keras.optimizers import SGD, Adam, RMSprop

# model = Sequential()
# model.add(Dense(128, input_dim=784, activation='linear')) # , kernel_regularizer=L2()
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dense(10, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=2, batch_size=500, validation_data=(X_test, y_test))

# y_pred = model.predict(X_test, batch_size=500, verbose=0)
# #print("Prediction:", np.round(y_pred.T))
# #print("Error:", np.abs(y_test - np.round(y_pred.T)), '%')
# print("Accuracy:", np.round((1 - np.sum(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1)) / X_test.shape[0]) * 100, 2), "%")

 
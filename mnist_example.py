from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
#np.seterr(all='ignore')

## Data
mnist = fetch_mldata('MNIST original')
img_rows, img_cols = 28, 28

classes = np.array(range(0, 10))
lb = preprocessing.LabelBinarizer()
lb.fit(classes)
mnist.target = lb.transform(mnist.target)

x_train, x_test, y_train, y_test = train_test_split(mnist.data/255, mnist.target, test_size=0.2)
#input_shape = img_rows * img_cols
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#input_shape = (img_rows, img_cols, 1)
#print(x_train.shape)
#print(y_train.shape)

## Neural Network
from ArtNet.models import Sequential
from ArtNet.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, Dropout, BatchNormalization
from ArtNet.activations import ReLU, Sigmoid, Softmax, Tanh, Leaky_ReLU, Linear
from ArtNet.regularizers import L2
from ArtNet.losses import Loss, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy
from ArtNet.optimizers import GradientDescent, RMSprop, Adam
from ArtNet.lib import r_squared

model = Sequential()
model.add(Input(input_shape=x_train.shape[1]))
#model.add(Input(input_shape=input_shape))
model.add(Dense(nodes=128, activation=ReLU(), weight_regularizer=L2()))
#model.add(Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation="ReLU", weight_initializer='GlorotUniform')) #
#model.add(Activation(ReLU()))
#model.add(MaxPooling2D(strides=(2, 2)))
#model.add(Flatten())
#model.add(Dense(nodes=128, activation=ReLU()))
model.add(Dense(nodes=10, activation=Softmax()))
model.compile(loss=CategoricalCrossEntropy(), optimizer=Adam(), metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=200, validation_data=(x_test, y_test))

#y_pred = model.predict(x_test, batch_size=500, verbose=0)
#print("Accuracy:", (1 - np.sum(np.abs(y_test - np.round(np.squeeze(y_pred)))) / y_test.shape[0]) * 100, '%')
#print("Accuracy:", np.round((1 - np.sum(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1)) / X_test.shape[0]) * 100, 2), "%")

## Keras
# from keras.models import Sequential
# from keras.layers import Dense, Activation, BatchNormalization, Conv2D, MaxPooling2D, Flatten
# from keras.optimizers import SGD, Adam, RMSprop

# model = Sequential()
# #model.add(Dense(128, input_dim=784, activation='tanh')) 
# model.add(Conv2D(32, input_shape=input_shape, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(strides=(2, 2)))
# model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(strides=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dense(10, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=2, batch_size=200, validation_data=(x_test, y_test))

# y_pred = model.predict(x_test, batch_size=500, verbose=0)
# print("Accuracy:", np.round((1 - np.sum(np.argmax(y_test, axis=1) != np.argmax(y_pred, axis=1)) / x_test.shape[0]) * 100, 2), "%")
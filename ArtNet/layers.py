import numpy as np
import importlib
from ArtNet.regularizers import Regularizer, L2

class Layer:

    n_layer = 0

    n_nodes = -1

    def __init__(self):
        pass
   
    def onAdd(self, model):
        model.n_layers += 1
        self.n_layer = model.n_layers
        
        model.W.append(0)
        model.b.append(0)  

        if self.n_layer > 0 and self.n_nodes == -1:
            self.n_nodes = model.layers[self.n_layer - 1].n_nodes

        model.layers.append(self)

    def onStart(self, model):
        pass

    def onEpochStart(self, model):
        pass

    def Forward(self, model, layer, A0):
        return A0

    def Backward(self, model, layer, dA):
        return dA

class Input(Layer):

    def __init__(self, features=1):
        self.n_nodes = features

class Dense(Layer):

    activation = None
    kernel_regularizer = None

    def __init__(self, nodes=1, activation="Linear", kernel_regularizer=None):
        self.n_nodes = nodes
        self.activation = activation
        self.kernel_regularizer = kernel_regularizer

    def onAdd(self, model):             
        if self.kernel_regularizer is not None:
            # adds weight regularization layer before this dense layer
            model.add(WeightRegularization(self.kernel_regularizer))

        super(Dense, self).onAdd(model)

        if self.activation is not None:
            # adds weight regularization layer after this dense layer
            model.add(Activation(activation=self.activation))

    def onStart(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].n_nodes
        model.W[self.n_layer] = np.random.randn(self.n_nodes, n_nodes0) * np.square(2.0 / self.n_nodes)
        model.b[self.n_layer] = np.zeros((self.n_nodes, 1))

    def Forward(self, model, layer, A0):
        return np.matmul(model.W[layer], A0) + model.b[layer]

    def Backward(self, model, layer, dA):
        model.dW[layer] = 1 / model.batch_size * np.matmul(dA, model.A[layer - 1].T)
        model.db[layer] = 1 / model.batch_size * np.sum(dA, keepdims=True)
        return np.matmul(model.W[layer].T, dA)

class BatchNormalization(Layer):

    momentum = 0.99
    epsilon = 1e-8

    mu = 0
    var = 0
    inv_var = 0
    normed = 0

    running_mu = 0
    running_var = 0

    def __init__(self, momentum=0.99, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon

    def onStart(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].n_nodes
        model.W[self.n_layer] = np.random.randn(self.n_nodes, n_nodes0) * np.square(2.0 / self.n_nodes)
        model.b[self.n_layer] = np.zeros((self.n_nodes, 1))

    def Forward(self, model, layer, A0):
        # Compute the exponetially weighted (running) average of the mean and variance
        if model.is_train:
            self.mu = np.mean(A0, axis=1, keepdims=True)
            self.var = np.var(A0, axis=1, keepdims=True)
            
            sqrt_var = np.sqrt(self.var + self.epsilon)
            self.inv_var = 1.0 / sqrt_var

            self.running_mu = self.momentum * self.running_mu + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            self.normed = (A0 - self.mu) / sqrt_var
        else:
            self.normed = (A0 - self.running_mu) / np.sqrt(self.running_var + self.epsilon)

        return np.matmul(model.W[layer], self.normed) + model.b[layer]

    def Backward(self, model, layer, dX0):

        dx_normed = np.matmul(model.W[layer].T, dX0)

        dX = (1.0 / model.batch_size) * self.inv_var * (model.batch_size * dx_normed - np.sum(dx_normed, axis=0) - self.normed * np.sum(dx_normed * self.normed, axis=0))

        model.dW[layer] = 1 / model.batch_size * np.matmul(dX0, self.normed.T)
        model.db[layer] = 1 / model.batch_size * np.sum(dX0, keepdims=True)

        return dX

class Activation(Layer):

    activation = None

    def __init__(self, activation="Linear"):
        if type(activation) == str:
            activation_class = getattr(importlib.import_module("ArtNet.activations"), activation)
            activation = activation_class()        
        self.activation = activation
   
    def Forward(self, model, layer, A0):
        return self.activation.g(A0)

    def Backward(self, model, layer, dA):
        return dA * self.activation.g_prime(model.A[layer - 1])

class Dropout(Layer):

    keep_prob = 0.8

    dropout = None

    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
   
    def onAdd(self, model):
        super(Dropout, self).onAdd(model)

    def onEpochStart(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].n_nodes
        self.dropout = np.random.randn(n_nodes0, model.batch_size) < self.keep_prob

    def Forward(self, model, layer, A0):
        if model.is_train:
            A = np.multiply(A0, self.dropout)
            A /= self.keep_prob
        else:
            A = A0

        return A

class WeightRegularization(Layer):

    regularizer = None

    def __init__(self, regularizer=None):
        # regularizer
        if type(self.regularizer) == str:
            regularizer_class = getattr(importlib.import_module("ArtNet.regularizers"), self.regularizer)
            regularizer = regularizer_class(regularizer_class)
        self.regularizer = regularizer
   
    def Backward(self, model, layer, dA):
        # Kernel regularization
        if isinstance(self.regularizer, L2):
            model.dW[layer - 1],  model.db[layer - 1] = self.regularizer.regularize(model.batch_size, model.W[layer - 1], model.dW[layer - 1], model.db[layer - 1])        
        return dA
import numpy as np
import importlib
from ArtNet.regularizers import Regularizer, L2

class Layer:

    #
    n_layer = 0
    n_nodes = -1

    # Output variable
    output = None

    # Defines if Layer is trainable meaning that whether the weights and biases can be optimized
    is_trainable = False

    # Weights and Biases
    weights = None
    dweights = None
    bias = None
    dbias = None   

    def __init__(self):
        pass
   
    def onAdd(self, model):
        model.n_layers += 1
        self.n_layer = model.n_layers

        if self.n_layer > 0 and self.n_nodes == -1:
            self.n_nodes = model.layers[self.n_layer - 1].n_nodes

        model.layers.append(self)

    def onStart(self, model):
        if self.is_trainable:
            n_nodes0 = model.layers[self.n_layer - 1].n_nodes
            self.weights = self.weight_initializer.initialize(n_nodes0, self.n_nodes)
            self.bias = self.bias_initializer.initialize(1, self.n_nodes)

    def onEpochStart(self, model):
        pass

    def Forward(self, model, layer):
        self.output = model.layers[layer - 1].output
        return self.output

    def Backward(self, model, layer, dOut0):
        return dOut0

class Input(Layer):

    def __init__(self, features=1):
        self.n_nodes = features

class Dense(Layer):

    is_trainable = True

    activation = None

    weight_initializer = ""
    bias_initializer = ""

    weight_regularizer = None

    def __init__(self, nodes=1, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
        self.n_nodes = nodes
        self.activation = activation

        if type(weight_initializer) == str:
            weight_initializer_class = getattr(importlib.import_module("ArtNet.initializers"), weight_initializer)
            weight_initializer = weight_initializer_class()     
        self.weight_initializer = weight_initializer

        if type(bias_initializer) == str:
            bias_initializer_class = getattr(importlib.import_module("ArtNet.initializers"), bias_initializer)
            bias_initializer = bias_initializer_class()     
        self.bias_initializer = bias_initializer

        self.weight_regularizer = weight_regularizer

    def onAdd(self, model):             
        if self.weight_regularizer is not None:
            # adds weight regularization layer before this dense layer
            model.add(WeightRegularization(self.weight_regularizer))

        super(Dense, self).onAdd(model)

        if self.activation is not None:
            # adds weight regularization layer after this dense layer
            model.add(Activation(activation=self.activation))

    def Forward(self, model, layer):
        self.output = np.matmul(self.weights, model.layers[layer - 1].output) + self.bias
        return self.output

    def Backward(self, model, layer, dOut0):
        self.dweights = 1 / model.batch_size * np.matmul(dOut0, model.layers[layer - 1].output.T)
        self.dbias = 1 / model.batch_size * np.sum(dOut0, keepdims=True)
        return np.matmul(self.weights.T, dOut0)

class Conv2D(Dense):

    kernel_size = 0
    strides = (1, 1)
    padding='valid'

    def __init__(self, nodes=1, kernel_size=1, strides=(1, 1), padding='valid', activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

        super(Conv2D, self).__init__(nodes=nodes, activation=activation, weight_initializer=weight_initializer, bias_initializer=bias_initializer, weight_regularizer=weight_regularizer)

class BatchNormalization(Layer):

    is_trainable = True

    momentum = 0.99
    epsilon = 1e-8

    mu = 0
    inv_var = 0
    normed = 0

    running_mu = 0
    running_var = 0

    def __init__(self, momentum=0.99, epsilon=1e-8, beta_initializer='Zeros', gamma_initializer='Ones'):
        self.momentum = momentum
        self.epsilon = epsilon

        if type(beta_initializer) == str:
            beta_initializer_class = getattr(importlib.import_module("ArtNet.initializers"), beta_initializer)
            beta_initializer = beta_initializer_class()     
        self.weight_initializer = beta_initializer

        if type(gamma_initializer) == str:
            gamma_initializer_class = getattr(importlib.import_module("ArtNet.initializers"), gamma_initializer)
            gamma_initializer = gamma_initializer_class()     
        self.bias_initializer = gamma_initializer        

    def onStart(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].n_nodes
        self.weights = self.weight_initializer.initialize(n_nodes0, self.n_nodes)
        self.bias = self.bias_initializer.initialize(1, self.n_nodes)

    def Forward(self, model, layer):
        a0 = model.layers[layer - 1].output
        # Compute the exponetially weighted (running) average of the mean and variance
        if model.is_train:
            self.mu = np.mean(a0, axis=1, keepdims=True)
            var = np.var(a0, axis=1, keepdims=True)
            
            sqrt_var = np.sqrt(var + self.epsilon)
            self.inv_var = 1.0 / sqrt_var

            self.running_mu = self.momentum * self.running_mu + (1 - self.momentum) * self.mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.normed = (a0 - self.mu) / sqrt_var
        else:
            self.normed = (a0 - self.running_mu) / np.sqrt(self.running_var + self.epsilon)
        self.output = np.matmul(self.weights, self.normed) + self.bias
        return self.output

    def Backward(self, model, layer, dOut0):
        dx_normed = np.matmul(self.weights.T, dOut0)
        dOut = (1.0 / model.batch_size) * self.inv_var * (model.batch_size * dx_normed - np.sum(dx_normed, axis=0) - self.normed * np.sum(dx_normed * self.normed, axis=0))
        self.dweights = 1 / model.batch_size * np.matmul(dOut0, self.normed.T)
        self.dbias = 1 / model.batch_size * np.sum(dOut0, keepdims=True)
        return dOut

class Activation(Layer):

    activation = None

    def __init__(self, activation="Linear"):
        if type(activation) == str:
            activation_class = getattr(importlib.import_module("ArtNet.activations"), activation)
            activation = activation_class()        
        self.activation = activation
   
    def Forward(self, model, layer):
        self.output = self.activation.g(model.layers[layer - 1].output)
        return self.output

    def Backward(self, model, layer, dA):
        return dA * self.activation.g_prime(model.layers[layer - 1].output)

class Dropout(Layer):

    keep_prob = 0.8

    seed = None

    dropout = None

    def __init__(self, keep_prob=0.8, seed=None):
        self.keep_prob = keep_prob
        self.seed = seed

    def onEpochStart(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].n_nodes
        np.random.seed(seed) if self.seed is not None else None
        self.dropout = np.random.randn(n_nodes0, model.batch_size) < self.keep_prob

    def Forward(self, model, layer):
        a0 = model.layers[layer - 1].output
        if model.is_train:
            A = np.multiply(a0, self.dropout)
            A /= self.keep_prob
        else:
            A = a0

        return A

class WeightRegularization(Layer):

    regularizer = None

    def __init__(self, regularizer=None):
        # regularizer
        if type(self.regularizer) == str:
            regularizer_class = getattr(importlib.import_module("ArtNet.regularizers"), self.regularizer)
            regularizer = regularizer_class(regularizer_class)
        self.regularizer = regularizer
   
    def Backward(self, model, layer, dOut0):
        # Weight regularization
        if isinstance(self.regularizer, L2) and model.layers[layer + 1].is_trainable:
            # regularize dweights of next layer
            model.layers[layer + 1].dweights,  model.layers[layer + 1].dbias = self.regularizer.regularize(model.batch_size, model.layers[layer + 1].weights, model.layers[layer + 1].dweights, model.layers[layer + 1].dbias)        
        return dOut0
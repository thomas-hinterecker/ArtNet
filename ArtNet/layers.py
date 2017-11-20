import numpy as np
import importlib
from ArtNet.regularizers import Regularizer, L2

class Layer:
    """Layer class"""

    shape = None
    n_layer = 0
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
        model.layers.append(self)

    def onStart(self, model):
        if self.shape is None:
            self.shape = model.layers[self.n_layer - 1].shape
        #
        if self.is_trainable:
            n_nodes0 = model.layers[self.n_layer - 1].shape[model.nodes_axis]
            self.weights = self.weight_initializer.initialize((self.shape[model.nodes_axis], n_nodes0))
            self.bias = self.bias_initializer.initialize((self.shape[model.nodes_axis], 1))
            assert(self.weights.shape == (self.shape[model.nodes_axis], n_nodes0))
            assert(self.bias.shape == (self.shape[model.nodes_axis], 1))

    def onEpochStart(self, model):
        pass

    def Forward(self, model, layer):
        self.output = model.layers[layer - 1].output
        return self.output

    def Backward(self, model, layer, dOut0):
        return dOut0

class Input(Layer):
    """Input layer"""

    def __init__(self, input_shape=1):
        if not isinstance(input_shape, (list, tuple)):
            input_shape = (input_shape,)
        self.shape = input_shape

    def onStart(self, model):
        if model.samples_axis == 0:
            self.shape = (model.batch_size,) + self.shape
        else:
            self.shape = self.shape + (model.batch_size,)

class Trainable(Layer):

    is_trainable = True

    def _InitLayer(self, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
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

class Dense(Trainable):
    """Dense neural network layer"""
    activation = None
    weight_initializer = None
    bias_initializer = None
    weight_regularizer = None

    def __init__(self, nodes, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
        self.shape = (nodes, )
        self._InitLayer(activation, weight_initializer, bias_initializer, weight_regularizer)

    def onAdd(self, model):             
        if self.weight_regularizer is not None:
            # adds weight regularization layer before this dense layer
            model.add(WeightRegularization(self.weight_regularizer))
        super(Dense, self).onAdd(model)
        if self.activation is not None:
            # adds weight regularization layer after this dense layer
            model.add(Activation(activation=self.activation))

    def onStart(self, model):
        if model.samples_axis == 0:
            self.shape = (model.batch_size,) + self.shape
        else:
            self.shape = self.shape + (model.batch_size,)
        super(Dense, self).onStart(model)

    def Forward(self, model, layer):
        self.output = (np.matmul(self.weights, model.layers[layer - 1].output) + self.bias)
        if model.is_train:
            assert(self.output.shape ==  self.shape)
        return self.output

    def Backward(self, model, layer, dOut0):
        self.dweights = 1 / model.batch_size * np.matmul(dOut0, model.layers[layer - 1].output.T)
        self.dbias = np.squeeze(1 / model.batch_size * np.sum(dOut0, keepdims=True))
        dOut = np.matmul(self.weights.T, dOut0)
        assert (dOut.shape == model.layers[self.n_layer - 1].shape)
        assert (self.dweights.shape == self.weights.shape)
        assert (isinstance(self.dbias, np.ndarray))
        return dOut

class Conv2D(Trainable):
    """2-D convolutional layer"""

    ##
    kernel_size_size = None
    strides = None
    padding = 'valid'
    paddings = None
    paddings_tuple = None
    data_format = None
    ##
    output_vertical = None
    output_horizontal = None

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
        self.shape = (filters, )
        self.kernel_size= kernel_size
        self.strides = strides
        self.padding = padding
        self._InitLayer(activation=activation, weight_initializer=weight_initializer, bias_initializer=bias_initializer, weight_regularizer=weight_regularizer)

    def onStart(self, model):
        input_shape = model.layers[self.n_layer - 1].shape

        self.paddings = np.zeros_like(self.kernel_size)
        if self.padding == 'same':
            self.paddings[0] = (self.kernel_size[0] - 1 * self.strides[0]) / 2
            self.paddings[1] = (self.kernel_size[1] - 1 * self.strides[1]) / 2
            if self.data_format == "channels_first":
                self.paddings_tuple = ((0, 0), (0, 0), (self.paddings[0], self.paddings[0]), (self.paddings[1], self.paddings[1]))
                vertical_size = input_shape[2]
                horizontal_size = input_shape[3]
                n_filters0 = input_shape[1]
            else:
                self.paddings_tuple = ((0, 0), (self.paddings[0], self.paddings[0]), (self.paddings[1], self.paddings[1]), (0, 0))
                vertical_size = input_shape[1]
                horizontal_size = input_shape[2]
                n_filters0 = input_shape[3]

        self.output_vertical = int(np.floor((vertical_size + 2 * self.paddings[0] - self.kernel_size[0]) / self.strides[0] + 1))
        self.output_horizontal = int(np.floor((horizontal_size + 2 * self.paddings[1] - self.kernel_size[1]) / self.strides[1] + 1))

        self.shape = (model.batch_size, self.output_vertical, self.output_horizontal, self.shape[-1])

        self.weights = self.weight_initializer.initialize((self.kernel_size[0], self.kernel_size[1], n_filters0, self.shape[-1]))
        self.bias = self.bias_initializer.initialize((1, 1, 1, self.shape[-1]))
        
    def Forward(self, model, layer):
        input = model.layers[layer - 1].output
        padded_input = np.pad(input, self.paddings_tuple, 'constant', constant_values=0)    
        self.output = np.zeros(self.shape)
        for sample in range(model.batch_size):                                 
            #padded_sample = padded_input[..., sample]
            for v in range(int(self.output_vertical / self.strides[0])):
                for h in range(int(self.output_horizontal / self.strides[1])): 
                    for f in range(self.shape[-1]):
                        # Find the corners of the current "slice"
                        vert_start = v * self.strides[0]
                        vert_end = vert_start + self.kernel_size[0]
                        horiz_start = h * self.strides[1]
                        horiz_end = horiz_start + self.kernel_size[1]
                        # Use the corners to define the (3D) slice of slice_sample
                        sliced_sample = padded_input[sample, vert_start:vert_end, horiz_start:horiz_end, :]
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                        self.output[sample, v, h, f] = self._StridedConvolution(sliced_sample, self.weights[..., f], self.bias[..., f])
        return self.output

    def _StridedConvolution(self, slice, W, b):
        return np.sum(np.multiply(slice, W) + b)

    def Backward(self, model, layer, dOut0):
        input = model.layers[layer - 1].output
        #
        dOut = np.zeros_like(input)
        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        #
        
        padded_input = np.pad(input, self.paddings_tuple, 'constant', constant_values=0)    
        padded_dOut = np.pad(dOut, self.paddings_tuple, 'constant', constant_values=0)    
        #
        for sample in range(model.batch_size):                                 
            for v in range(int(self.output_vertical / self.strides[0])):
                for h in range(int(self.output_horizontal / self.strides[1])): 
                    for f in range(self.shape[-1]):
                        # Find the corners of the current "slice"
                        vert_start = v * self.strides[0]
                        vert_end = vert_start + self.kernel_size[0]
                        horiz_start = h * self.strides[1]
                        horiz_end = horiz_start + self.kernel_size[1]                        
                        #
                        sliced_sample = padded_input[sample, vert_start:vert_end, horiz_start:horiz_end, :]
                        # Update gradients for the window and the filter's parameters
                        padded_dOut[sample, vert_start:vert_end, horiz_start:horiz_end, :] += self.weights[:, :, :, f] * dOut0[sample, v, h, f]
                        self.dweights[:, :, :, f] += sliced_sample * dOut0[sample, v, h, f]
                        self.dbias[:, :, :, f] += dOut0[sample, v, h, f]
            # Set to unpaded data
            dOut[sample, :, :, :] = padded_dOut[sample, self.paddings[0]:-self.paddings[0], self.paddings[1]:-self.paddings[1], :]

        return dOut

class MaxPooling2D(Layer):

    pool_size = None
    strides = None
    padding = None
    data_format = None

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', data_format=None):
        self.pool_size = pool_size
        self.strides= strides
        self.padding = padding
        self.data_format = data_format

    def onStart(self, model):
        input_shape = model.layers[self.n_layer - 1].shape

        paddings = np.zeros_like(self.pool_size)
        if self.data_format == "channels_first":
            vertical_size = input_shape[2]
            horizontal_size = input_shape[3]
            n_filters0 = input_shape[1]
        else:
            vertical_size = input_shape[1]
            horizontal_size = input_shape[2]
            n_filters0 = input_shape[3]

        self.output_vertical = int(np.floor((vertical_size + 2 * paddings[0] - self.pool_size[0]) / self.strides[0] + 1))
        self.output_horizontal = int(np.floor((horizontal_size + 2 * paddings[1] - self.pool_size[1]) / self.strides[1] + 1))

        self.shape = (model.batch_size, self.output_vertical, self.output_horizontal, n_filters0)
        
    def Forward(self, model, layer):
        input = model.layers[layer - 1].output
        self.output = np.zeros(self.shape)
        for sample in range(model.batch_size):                                 
            for v in range(int(self.output_vertical / self.strides[0])):
                for h in range(int(self.output_horizontal / self.strides[1])): 
                    for f in range(self.shape[-1]):
                        # Find the corners of the current "slice"
                        vert_start = v * self.strides[0]
                        vert_end = vert_start + self.pool_size[0]
                        horiz_start = h * self.strides[1]
                        horiz_end = horiz_start + self.pool_size[1]
                        # Use the corners to define the (3D) slice
                        slice_sample = input[sample, vert_start:vert_end, horiz_start:horiz_end, f]
                        # Max pooling
                        self.output[sample, v, h, f] = np.max(slice_sample)
        return self.output

    def Backward(self, model, layer, dOut0):
        dOut = np.zeros(model.layers[self.n_layer - 1].shape)
        for sample in range(model.batch_size):                                 
            for v in range(int(self.output_vertical / self.strides[0])):
                for h in range(int(self.output_horizontal / self.strides[1])): 
                    for f in range(self.shape[-1]): 
                        # Find the corners of the current "slice"
                        vert_start = v * self.strides[0]
                        vert_end = vert_start + self.pool_size[0]
                        horiz_start = h * self.strides[1]
                        horiz_end = horiz_start + self.pool_size[1]       
                        # Use the corners and "c" to define the current slice from a_prev
                        slice_sample = dOut0[sample, vert_start:vert_end, horiz_start:horiz_end, f]
                        mask = self._CreateMaskFromWindow(slice_sample)
                        dOut[sample, vert_start:vert_end, horiz_start:horiz_end, f] += np.multiply(mask, dOut0[sample, v, h, f])  
        return dOut

    def _CreateMaskFromWindow(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        mask = x == np.max(x)
        return mask

    # def _DistributeValue(dz, shape):
    #     """
    #     Distributes the input value in the matrix of dimension shape
        
    #     Arguments:
    #     dz -- input scalar
    #     shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
    #     Returns:
    #     a -- Array of size (n_H, n_W) for which we distributed the value of dz
    #     """
    #     # Retrieve dimensions from shape
    #     (n_H, n_W) = shape
    #     # Compute the value to distribute on the matrix (≈1 line)
    #     average = dz / (n_H * n_W)
    #     # Create a matrix where every entry is the "average" value (≈1 line)
    #     return np.ones(shape) * average

class Flatten(Layer):

    def onStart(self, model):
        input_shape = model.layers[self.n_layer - 1].shape
        n_nodes = 1
        for i in range(1, len(input_shape)):
            n_nodes *= input_shape[i]
        self.shape = (model.batch_size, n_nodes)

    def Forward(self, model, layer):
        input = model.layers[self.n_layer - 1].output
        self.output = input.reshape(self.shape)
        return self.output

    def Backward(self, model, layer, dOut0):
        return dOut0.reshape(model.layers[self.n_layer - 1].shape)

class BatchNormalization(Layer):
    """Batch normalization layer"""

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
    """Activation layer"""

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
    """Dropout layer"""    

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
    """Weight regularization"""

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
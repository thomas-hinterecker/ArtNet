import importlib

import numpy as np

from ArtNet.regularizers import L2


class Layer:
    """Layer class"""

    output_shape = None
    n_layer = 0
    output = None
    is_trainable = False
    weights = None
    dweights = None
    bias = None
    dbias = None
    weight_initializer = None
    bias_initializer = None
    paddings = None

    def __init__(self):
        pass

    def on_add(self, model):
        model.n_layers += 1
        self.n_layer = model.n_layers
        model.layers.append(self)

    def on_start(self, model):
        if self.output_shape is None:
            self.output_shape = model.layers[self.n_layer - 1].output_shape
        if self.is_trainable and model.is_train:
            n_nodes0 = model.layers[self.n_layer - 1].output_shape[model.nodes_axis]
            self.weights = self.weight_initializer.initialize((self.output_shape[model.nodes_axis], n_nodes0))
            self.bias = self.bias_initializer.initialize((self.output_shape[model.nodes_axis], 1))
            assert (self.weights.shape == (self.output_shape[model.nodes_axis], n_nodes0))
            assert (self.bias.shape == (self.output_shape[model.nodes_axis], 1))

    def on_epoch_start(self, model):
        pass

    def forward(self, model, layer):
        self.output = model.layers[layer - 1].output
        return self.output

    def backward(self, model, layer, d_out0):
        return d_out0


class Input(Layer):
    """Input layer"""

    def __init__(self, input_shape=1):
        super(Input, self).__init__()
        self.output_shape = (input_shape,) if not isinstance(input_shape, (list, tuple)) else tuple(
            reversed(input_shape))

    def on_start(self, model):
        self.output_shape = self.output_shape + (model.batch_size,)


class Trainable(Layer):
    is_trainable = True

    def _init_layer(self, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros',
                    weight_regularizer=None):
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

    def __init__(self, nodes, activation="Linear", weight_initializer='GlorotUniform', bias_initializer='Zeros',
                 weight_regularizer=None):
        super(Dense, self).__init__()
        self.output_shape = (nodes,)
        self._init_layer(activation, weight_initializer, bias_initializer, weight_regularizer)

    def on_add(self, model):
        if self.weight_regularizer is not None:
            # adds weight regularization layer before this dense layer
            model.add(WeightRegularization(self.weight_regularizer))
        super(Dense, self).on_add(model)
        if self.activation is not None:
            # adds weight regularization layer after this dense layer
            model.add(Activation(activation=self.activation))

    def on_start(self, model):
        self.output_shape = self.output_shape + (model.batch_size,)
        super(Dense, self).on_start(model)

    def forward(self, model, layer):
        self.output = np.matmul(
            self.weights, model.layers[layer - 1].output) + self.bias
        if model.is_train:
            assert (self.output.shape == self.output_shape)
        return self.output

    def backward(self, model, layer, d_out0):
        self.dweights = 1 / model.batch_size * \
                        np.matmul(d_out0, model.layers[layer - 1].output.T)
        self.dbias = np.squeeze(1 / model.batch_size *
                                np.sum(d_out0, keepdims=True))
        d_out = np.matmul(self.weights.T, d_out0)
        assert (d_out.shape == model.layers[self.n_layer - 1].output_shape)
        assert (self.dweights.shape == self.weights.shape)
        assert (isinstance(self.dbias, np.ndarray))
        return d_out


class Conv2D(Trainable):
    """2-D convolutional layer"""

    kernel_size = None
    strides = None
    padding = 'valid'
    paddings = None
    paddings_tuple = None
    data_format = None
    output_height = None
    output_width = None
    output_channels = None
    input_data = None
    input_data_column_vector = None

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation="Linear",
                 weight_initializer='GlorotUniform', bias_initializer='Zeros', weight_regularizer=None):
        super(Conv2D, self).__init__()
        self.output_shape = (filters,)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self._init_layer(activation=activation, weight_initializer=weight_initializer,
                         bias_initializer=bias_initializer, weight_regularizer=weight_regularizer)

    def on_start(self, model):
        input_channels, input_width, input_height, m = model.layers[self.n_layer - 1].output_shape
        # set paddings
        self.paddings = np.zeros_like(self.kernel_size)
        if self.padding == 'same':
            self.paddings[0] = (self.kernel_size[0] - 1 * self.strides[0]) / 2
            self.paddings[1] = (self.kernel_size[1] - 1 * self.strides[1]) / 2
        self.paddings_tuple = (
            (0, 0), (self.paddings[0], self.paddings[0]), (self.paddings[1], self.paddings[1]), (0, 0))
        # get output dimensions
        assert (input_height + 2 * self.paddings[0] - self.kernel_size[0]) % self.strides[0] == 0
        assert (input_width + 2 * self.paddings[1] - self.kernel_size[1]) % self.strides[1] == 0
        self.output_height = int(
            np.floor((input_height + 2 * self.paddings[0] - self.kernel_size[0]) / self.strides[0] + 1))
        self.output_width = int(
            np.floor((input_width + 2 * self.paddings[1] - self.kernel_size[1]) / self.strides[1] + 1))
        self.output_channels = self.output_shape[0]
        # set output shape
        self.output_shape = (self.output_channels, self.output_width, self.output_height, m)
        # initialize weight and bias matrices
        self.weights = self.weight_initializer.initialize(
            (self.output_channels, input_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = self.bias_initializer.initialize((self.output_channels, 1))

    def forward(self, model, layer):
        self.input_data = model.layers[layer - 1].output
        self.input_data_column_vector = self.matrix2_column(self.input_data.T)
        weights_colum_vector = self.weights.reshape(self.output_channels, -1)
        out = np.matmul(weights_colum_vector, self.input_data_column_vector) + self.bias
        self.output = out.reshape((self.output_shape[0], self.output_shape[1], self.output_shape[2], model.batch_size))
        return self.output

    def backward(self, model, layer, d_out0):
        d_out0 = d_out0.T
        self.dbias = np.sum(d_out0, axis=(0, 1, 2))
        self.dbias = self.dbias.reshape(self.output_channels, -1)
        d_out0_reshaped = d_out0.transpose(3, 1, 2, 0).reshape(self.output_channels, -1)
        self.dweights = np.matmul(d_out0_reshaped, self.input_data_column_vector.T)
        self.dweights = self.dweights.reshape(self.weights.shape)
        weights_reshaped = self.weights.reshape(self.output_channels, -1)
        d_out_col = np.matmul(weights_reshaped.T, d_out0_reshaped)
        out = self.column2_matrix(d_out_col, self.input_data.shape)
        return out

    def matrix2_column(self, x):
        x_padded = np.pad(x, self.paddings_tuple, mode='constant')
        k, i, j = self.matrix2_column_indices(x.shape)
        cols = x_padded[:, j, i, k]
        cols = cols.transpose(1, 2, 0).reshape(
            self.kernel_size[0] * self.kernel_size[1] * x.shape[-1], -1)
        return cols

    def matrix2_column_indices(self, x_shape):
        n, w, h, c = x_shape
        out_height = int((h + 2 * self.paddings[0] - self.kernel_size[0]) / self.strides[0] + 1)
        out_width = int((w + 2 * self.paddings[1] - self.kernel_size[1]) / self.strides[1] + 1)
        i0 = np.repeat(np.arange(self.kernel_size[0]), self.kernel_size[1])
        i0 = np.tile(i0, c)
        i1 = self.strides[0] * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(self.kernel_size[1]), self.kernel_size[0] * c)
        j1 = self.strides[1] * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(c), self.kernel_size[0] * self.kernel_size[1]).reshape(-1, 1)
        return k, i, j

    def column2_matrix(self, cols, x_shape):
        n, h, w, c = x_shape
        h_padded, w_padded = h + 2 * self.paddings[0], w + 2 * self.paddings[1]
        x_padded = np.zeros((n, h_padded, w_padded, c), dtype=cols.dtype)
        k, i, j = self.matrix2_column_indices(x_shape)
        cols_reshaped = cols.reshape(self.kernel_size[0] * self.kernel_size[1] * c, -1, n)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), j, i, k), cols_reshaped)
        if self.paddings[0] == 0 and self.paddings[1] == 0:
            return x_padded
        return x_padded[:, self.paddings[0]:-self.paddings[0], self.paddings[1]:-self.paddings[1], :]


class MaxPooling2D(Conv2D):
    max_idx = None

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid'):
        super(Conv2D, self).__init__()
        self.kernel_size = pool_size
        self.strides = strides
        self.padding = padding

    def on_start(self, model):
        input_channels, input_width, input_height, m = model.layers[self.n_layer - 1].output_shape
        self.paddings = np.zeros_like(self.kernel_size)
        self.paddings_tuple = (
            (0, 0), (0, 0), (self.paddings[0], self.paddings[0]), (self.paddings[1], self.paddings[1]))
        # get output dimensions
        assert (input_height + 2 * self.paddings[0] - self.kernel_size[0]) % self.strides[0] == 0
        assert (input_width + 2 * self.paddings[1] - self.kernel_size[1]) % self.strides[1] == 0
        self.output_height = int(
            np.floor((input_height + 2 * self.paddings[0] - self.kernel_size[0]) / self.strides[0] + 1))
        self.output_width = int(
            np.floor((input_width + 2 * self.paddings[1] - self.kernel_size[1]) / self.strides[1] + 1))
        self.output_channels = input_channels
        self.output_shape = (self.output_channels, self.output_width, self.output_height, m)

    def forward(self, model, layer):
        self.input_data = model.layers[layer - 1].output
        # gen_image(self.input_data[0,:,:,0]).show()
        # Reshape it to make im2col arranges it fully in column
        input_data_reshaped = self.input_data.T.reshape(self.input_data.shape[-1] * self.input_data.shape[0],
                                                        self.input_data.shape[1], self.input_data.shape[2], 1)
        # gen_image(input_data_reshaped.T[0,:,:,0]).show()
        self.input_data_column_vector = self.matrix2_column(input_data_reshaped)
        # Next, at each possible patch location, i.e. at each column, we're taking the max index
        self.max_idx = np.argmax(self.input_data_column_vector, axis=0)
        # Finally, we get all the max value at each column
        out = self.input_data_column_vector[self.max_idx, range(self.max_idx.size)]
        # Reshape to the output size
        out = out.reshape((self.output_shape[2], self.output_shape[1], model.batch_size, self.output_shape[0]))
        self.output = out.transpose(3, 0, 1, 2)
        # gen_image(self.output[0,:,:,0], shape=(14, 14)).show()
        return self.output

    def backward(self, model, layer, d_out0):
        # gen_image(dOut0[0,:,:,0], shape=(14, 14)).show()
        d_out_column_vector = np.zeros_like(self.input_data_column_vector)
        # Transpose step is necessary to get the correct arrangement
        d_out0_flat = d_out0.transpose(1, 2, -1, 0).ravel()
        # Fill the maximum index of each column with the gradient
        d_out_column_vector[self.max_idx, range(self.max_idx.size)] = d_out0_flat
        # We now have the stretched matrix of 4x9800, then undo it with col2im operation
        d_out = self.column2_matrix(d_out_column_vector, (
            self.input_data.shape[-1] * self.input_data.shape[0], self.input_data.shape[1], self.input_data.shape[2],
            1))
        # Reshape back to match the input dimension
        d_out = d_out.transpose((0, -1, 1, 2)).reshape((self.input_data.shape[-1], self.input_data.shape[0],
                                                        self.input_data.shape[1], self.input_data.shape[2])).transpose(
            (1, 3, 2, 0))
        # gen_image(dOut[0,:,:,0]).show()
        return d_out


class Flatten(Layer):

    def on_start(self, model):
        input_shape = model.layers[self.n_layer - 1].output_shape
        n_nodes = 1
        for i in range(0, len(input_shape) - 1):
            n_nodes *= input_shape[i]
        self.output_shape = (n_nodes, model.batch_size)

    def forward(self, model, layer):
        input_data = model.layers[self.n_layer - 1].output
        self.output = input_data.reshape((self.output_shape[0], -1))
        return self.output

    def backward(self, model, layer, d_out0):
        return d_out0.reshape(model.layers[self.n_layer - 1].output_shape)


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
        super(BatchNormalization, self).__init__()
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

    def forward(self, model, layer):
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

    def backward(self, model, layer, d_out0):
        dx_normed = np.matmul(self.weights.T, d_out0)
        d_out = (1.0 / model.batch_size) * self.inv_var * (
                model.batch_size * dx_normed - np.sum(dx_normed, axis=0) - self.normed * np.sum(dx_normed * self.normed,
                                                                                                axis=0))
        self.dweights = 1 / model.batch_size * np.matmul(d_out0, self.normed.T)
        self.dbias = 1 / model.batch_size * np.sum(d_out0, keepdims=True)
        return d_out


class Activation(Layer):
    """Activation layer"""

    activation = None

    def __init__(self, activation="Linear"):
        super(Activation, self).__init__()
        if type(activation) == str:
            activation_class = getattr(importlib.import_module("ArtNet.activations"), activation)
            activation = activation_class()
        self.activation = activation

    def forward(self, model, layer):
        self.output = self.activation.g(model.layers[layer - 1].output)
        return self.output

    def backward(self, model, layer, d_a):
        return d_a * self.activation.g_prime(model.layers[layer - 1].output)


class Dropout(Layer):
    """Dropout layer"""

    keep_prob = 0.8

    seed = None

    dropout = None

    def __init__(self, keep_prob=0.8, seed=None):
        super(Dropout, self).__init__()
        self.keep_prob = keep_prob
        self.seed = seed

    def on_epoch_start(self, model):
        n_nodes0 = model.layers[self.n_layer - 1].output_shape[0]
        np.random.seed(self.seed) if self.seed is not None else None
        self.dropout = np.random.randn(
            n_nodes0, model.batch_size) < self.keep_prob

    def forward(self, model, layer):
        input_data = model.layers[layer - 1].output
        if model.is_train:
            out = np.multiply(input_data, self.dropout)
            out /= self.keep_prob
        else:
            out = input_data
        self.output = out
        return out


class WeightRegularization(Layer):
    """Weight regularization"""

    regularizer = None

    def __init__(self, regularizer=None):
        super(WeightRegularization, self).__init__()
        if type(self.regularizer) == str:
            regularizer_class = getattr(importlib.import_module("ArtNet.regularizers"), self.regularizer)
            regularizer = regularizer_class(regularizer_class)
        self.regularizer = regularizer

    def backward(self, model, layer, d_out0):
        # Weight regularization
        if isinstance(self.regularizer, L2) and model.layers[layer + 1].is_trainable:
            # regularize dweights of next layer
            model.layers[layer + 1].dweights, model.layers[layer + 1].dbias = self.regularizer.regularize_dweights(
                model.batch_size, model.layers[layer + 1].weights, model.layers[layer + 1].dweights,
                model.layers[layer + 1].dbias)
        return d_out0

    def loss_regularization(self, model, layer):
        out = 0
        if isinstance(self.regularizer, L2) and model.layers[layer + 1].is_trainable:
            out = self.regularizer.regularize_loss(
                model.layers[layer + 1].weights)
        return out

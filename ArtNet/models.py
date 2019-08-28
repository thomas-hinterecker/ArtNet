import importlib
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from ArtNet.layers import Layer, WeightRegularization
from ArtNet.lib import print_progress_bar


class Sequential:
    """ Neural Network Model """

    samples_axis = -1
    nodes_axis = 0
    epoch = 0
    batch_size = 32
    is_train = False
    layers = []
    n_layers = -1
    lossf = None
    optimizer = None
    metrics = None

    def __init__(self):
        pass

    def add(self, obj):
        # if object is of type layer
        if isinstance(obj, Layer):
            obj.on_add(self)

    def compile(self, loss="MeanSquaredError", optimizer="GradientDescent", metrics=None):
        # Loss function
        if type(loss) == str:
            loss_function_class = getattr(
                importlib.import_module("ArtNet.losses"), loss)
            loss = loss_function_class()
        self.lossf = loss
        # Optimizer
        if type(optimizer) == str:
            optimizer_class = getattr(
                importlib.import_module("ArtNet.optimizers"), optimizer)
            optimizer = optimizer_class()
        self.optimizer = optimizer
        # Metrics
        self.metrics = metrics

    def fit(self, x, y, epochs=1, batch_size=32, verbose=1, validation_data=None):
        self.is_train = True
        self.batch_size = batch_size
        x, y = self._prep_data(x, y)
        n_samples = x.shape[self.samples_axis]
        [self.layers[layer].on_start(self) for layer in range(0, self.n_layers + 1)]
        # batch train
        num_batches = int(n_samples / self.batch_size)
        test_x_batches = self._prep_batches(x, num_batches, axis=self.samples_axis)
        test_y_batches = self._prep_batches(y, num_batches, axis=self.samples_axis)
        # Initialize the optimizer
        self.optimizer.init(self)
        # loop over epochs
        start_time = 0
        difftime = 0
        prefix = suffix = ''
        for epoch in range(0, epochs):
            self.epoch = epoch
            [self.layers[layer].on_epoch_start(self) for layer in range(1, self.n_layers + 1)]
            if verbose == 1:
                print('Epoch: ' + str(epoch + 1) + '/' + str(epochs)) if epoch > 0 else None
                start_time = timer()
                print_progress_bar(0, num_batches,
                                   prefix='{:{width}.0f}'.format(0, width=len(str(n_samples))) + '/' + str(
                                       n_samples) + ' -', suffix='- loss:', length=np.min((30, num_batches)),
                                   timer=int(start_time))
            # loop over batches
            loss = 1
            i = 0
            for i in range(0, num_batches):
                output = self._do_forward_prop(test_x_batches[i])
                # compute loss
                if np.round(i % (num_batches * .05)) == 0:
                    loss = self._compute_loss(test_y_batches[i])
                self._do_backward_prop(test_y_batches[i])
                if verbose == 1:
                    difftime = int(np.round(timer() - start_time, 0))
                    prefix = '{:{width}.0f}'.format(
                        self.batch_size * (i + 1), width=len(str(n_samples))) + '/' + str(n_samples) + ' -'
                    suffix = '- loss: {:.4f}'.format(np.round(loss, 4))
                    if 'accuracy' in self.metrics:
                        suffix += ' - acc: {:1.4f}'.format(np.round(self.lossf.accuracy(output, test_y_batches[i]), 4))
                    print_progress_bar(i + 1, num_batches, prefix=prefix, suffix=suffix + (' ' * 5),
                                       length=np.min((30, num_batches)), timer=difftime)
            # Learning rate decay
            self.optimizer.lr_decay(self)
            # Validation
            if validation_data is not None:
                if verbose == 1:
                    val_loss = self.evaluate(validation_data[0], validation_data[1],
                                             batch_size=validation_data[0].shape[self.nodes_axis])
                    self.batch_size = batch_size
                    suffix += ' - val_loss: {:.4f}'.format(np.round(val_loss, 4))
                    if 'accuracy' in self.metrics:
                        suffix += ' - val_acc: {:1.4f}'.format(
                            np.round(self.lossf.accuracy(self.layers[self.n_layers].output, validation_data[1].T), 4))
                    print_progress_bar(i + 1, num_batches, prefix=prefix, suffix=suffix,
                                       length=np.min((30, num_batches)), timer=difftime) if verbose == 1 else None
            print() if verbose == 1 else None

    def evaluate(self, x=None, y=None, batch_size=None):
        self.is_train = False
        self.batch_size = batch_size
        x, y = self._prep_data(x, y)
        # batches
        num_batches = int(x.shape[self.samples_axis] / self.batch_size)
        val_x_batches = self._prep_batches(x, num_batches, axis=self.samples_axis)
        val_y_batches = self._prep_batches(y, num_batches, axis=self.samples_axis)
        # loop over batches
        for i in range(0, num_batches):
            self._do_forward_prop(val_x_batches[i])
        return self._compute_loss(val_y_batches[i])

    def predict(self, x, batch_size=32, verbose=0):
        self.is_train = False
        self.batch_size = batch_size
        x, y = self._prep_data(x)
        n_samples = x.shape[self.samples_axis]
        # batches
        num_batches = int(n_samples / self.batch_size)
        x_batches = self._prep_batches(X, num_batches, axis=self.samples_axis)
        y_pred = []
        start_time = 0
        if verbose == 1:
            start_time = timer()
            print_progress_bar(0, num_batches, prefix='0/' + str(n_samples), suffix='', length=30,
                               timer=int(start_time))
        # loop over batches
        for i in range(0, num_batches):
            output = self._do_forward_prop(x_batches[i])
            y_pred.append(output.T)
            print_progress_bar(i + 1, num_batches, prefix=str(self.batch_size * (i + 1)) + '/' + str(n_samples) + ' -',
                               suffix='', length=np.min((30, num_batches)),
                               timer=int(np.round(timer() - start_time, 0))) if verbose == 1 else None
        print() if verbose == 1 else None
        return np.squeeze(np.array(y_pred).reshape(x.shape[1], self.a[self.n_layers].shape[0]))

    def _prep_data(self, x, y=None):
        # Check if input is a pandas dataframe. If so, convert to np.array
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = np.array(x.values)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = np.array(y.values)
        # Tanspose input so that num of features is across rows
        x = x.T
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], 1)
            y = y.T
        return x, y

    def _prep_batches(self, data, num_batches, axis=-1):
        """Split data into batches"""
        return np.split(data, num_batches, axis=axis)

    def _compute_loss(self, y):
        loss = self.lossf.L(self.layers[self.n_layers].output, y)
        # loss regularization
        regularization = 0
        for layer in range(1, self.n_layers + 1):
            if isinstance(self.layers[layer], WeightRegularization):
                regularization += self.layers[layer].loss_regularization(self, layer)
        regularization = 1 / self.batch_size * regularization
        loss += regularization
        return loss

    def _do_forward_prop(self, a0):
        self.layers[0].output = a0
        for layer in range(1, self.n_layers + 1):
            self.layers[layer].forward(self, layer)
        return self.layers[self.n_layers].output

    def _do_backward_prop(self, y):
        # compute dA[L]
        da = self.lossf.L_prime(self.layers[self.n_layers].output, y)
        # do back-prop
        for layer in range(self.n_layers, 0, -1):
            da = self.layers[layer].backward(self, layer, da)
        # adjust weight matrix
        self.optimizer.optimize(self)

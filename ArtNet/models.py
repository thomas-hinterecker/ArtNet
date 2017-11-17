import numpy as np
import pandas as pd
import importlib
from timeit import default_timer as timer
from ArtNet.lib import printProgressBar
from ArtNet.layers import Layer, Input
from ArtNet.regularizers import Regularizer
from ArtNet.optimizers import Optimizer

class Sequential:
    """ Neural Network Model """

    module_name = "ArtNet"

    ##
    epoch = 0
    batch_size = 32
    is_train = False

    ## Layers
    layers = []
    n_layers = -1
    
    ##
    lossf = None
    optimizer = None
    metrics = None

    ##
    A = []
    W = []
    b = []
    dW = []
    db = []
        
    def __init__(self):
        self.W.append(0)
        self.b.append(0)

    def add(self, object):
        # if object is of type layer
        if isinstance(object, Layer):
            object.onAdd(self)

    def compile(self, loss="logistic_regression", optimizer="GradientDescent", metrics=None):
        # Loss function
        if type(loss) == str:
            loss_function_class = getattr(importlib.import_module(self.module_name + ".losses"), loss)
            loss = loss_function_class()
        self.lossf = loss

        # Optimizer
        if type(optimizer) == str:
            optimizer_class = getattr(importlib.import_module(self.module_name + ".optimizers"), optimizer)
            optimizer = optimizer_class()
        self.optimizer = optimizer

        self.metrics = metrics

    def fit(self, X, y, epochs=1, batch_size=32, verbose=1, validation_data=None):
        self.is_train = True
        self.batch_size = batch_size
        X, y = self._prepData(X, y)
        
        [self.layers[layer].onStart(self) for layer in range(1, self.n_layers + 1)]
        
        # batch train
        num_batches = int(X.shape[1] / self.batch_size)
        test_x_batches = self._prepBatches(X, num_batches, axis=1)
        test_y_batches = self._prepBatches(y, num_batches, axis=1)
        
        # Initialize the optimizer
        self.optimizer.init(self)
        
        # loop over epochs
        for epoch in range(0, epochs):
            self.epoch = epoch
            [self.layers[layer].onEpochStart(self) for layer in range(1, self.n_layers + 1)]
            
            # print
            if verbose == 1:
                print('Epoch: ' + str(epoch + 1) + '/' + str(epochs)) if epoch > 0 else None
                start_time = timer()
                printProgressBar(0, num_batches, prefix = '0/' + str(X.shape[1]) + ' -', suffix = '- loss: ', length = 30, timer = int(start_time))
            
            # loop over batches
            loss = 1
            for i in range(0, num_batches):
                
                self._doForwardProp(test_x_batches[i])
                
                # compute loss
                if np.round(i % (num_batches * .05)) == 0:
                    loss = self._computeLoss(test_y_batches[i])
                
                self._doBackwardProp(test_y_batches[i])
                
                # print
                prefix = str(self.batch_size * (i + 1)) + '/' + str(X.shape[1]) + ' -'
                suffix = '- loss: ' + str(np.round(loss, 4))
                difftime = int(np.round(timer() - start_time, 0))
                printProgressBar(i + 1, num_batches, prefix = prefix, suffix = suffix, length = 30, timer = difftime) if verbose == 1 else None
            
            # Learning rate decay
            self.optimizer.lr_decay(self)

            # Validation
            if validation_data is not None:
                val_loss = self.evaluate(validation_data[0], validation_data[1], self.batch_size)
                suffix = suffix + ' - val_loss: ' + str(np.round(val_loss, 4))
                
                if 'accuracy' in self.metrics:   
                    y_pred = self.predict(validation_data[0], self.batch_size)
                    accuracy = np.round(1 - np.sum(np.argmax(validation_data[1], axis=1) != np.argmax(y_pred, axis=1)) / validation_data[0].shape[0], 4)
                    suffix = suffix + ' - val_acc: ' + str(accuracy)

                # print
                printProgressBar(i + 1, num_batches, prefix = prefix, suffix = suffix, length = 30, timer = difftime) if verbose == 1 else None
                
            print()

    def evaluate(self, X=None, y=None, batch_size=None, verbose=1):
        self.is_train = False
        self.batch_size = batch_size
        X, y = self._prepData(X, y)
        self.layers[0].n_nodes = X.shape[0]

        # batches
        num_batches = int(X.shape[1] / self.batch_size)
        val_x_batches = self._prepBatches(X, num_batches, axis=1)
        val_y_batches = self._prepBatches(y, num_batches, axis=1)
        
        # loop over batches
        for i in range(0, num_batches):
            self._doForwardProp(val_x_batches[i])
        loss = self._computeLoss(val_y_batches[i])

        return loss

    def predict(self, X, batch_size=32, verbose=0):
        self.is_train = False
        self.batch_size = batch_size
        X, y = self._prepData(X)
        self.layers[0].n_nodes = X.shape[0]
        
        # batches
        num_batches = int(X.shape[1] / self.batch_size)
        x_batches = self._prepBatches(X, num_batches, axis=1)
        y_pred = []
        
        # print
        if verbose == 1:
            start_time = timer()
            printProgressBar(0, num_batches, prefix = '0/' + str(X.shape[1]), suffix = '', length = 30, timer = int(start_time))
        
        # loop over batches
        for i in range(0, num_batches):
            self._doForwardProp(x_batches[i])
            y_pred.append(self.A[self.n_layers].T)
            
            # print
            printProgressBar(i + 1, num_batches, prefix = str(self.batch_size * (i + 1)) + '/' + str(X.shape[1]) + ' -', suffix = '', length = 30, timer = int(np.round(timer() - start_time, 0))) if verbose == 1 else None
        print() if verbose == 1 else None

        return np.squeeze(np.array(y_pred).reshape(X.shape[1], self.A[self.n_layers].shape[0]))

    def _prepData(self, X, y = None):
        # Check if input is a pandas dataframe. If so, convert to np.array
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = np.array(X.values)
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = np.array(y.values)
        
        # Tanspose input so that num of features is across rows
        X = X.T
        if y is not None:
            if y.ndim == 1:
                y = y.reshape(y.shape[0], 1)
            y = y.T
        
        return X, y

    def _prepBatches(self, data, num_batches, axis=1):
        # split data into batches
        return np.split(data, num_batches, axis=axis)  

    def _computeLoss(self, y):
        return self.lossf.L(self.A[self.n_layers], y)

    def _doForwardProp(self, a0):
        # init vars for forward-prop
        self.A = [a0]
        [self.A.append(None) for i in range(-1, self.n_layers)]
        
        # do forward-prop
        A0 = self.A[0]
        for layer in range(1, self.n_layers + 1):
            self.A[layer] = a0 = self.layers[layer].Forward(self, layer, a0)

    def _doBackwardProp(self, y):
        # init vars for backward-prop
        self.dW = [0]; self.db = [0]
        [[self.dW.append(None), self.db.append(None)] for i in range(-1, self.n_layers)]
        
        # compute dA[L]
        dA = self.lossf.L_prime(self.A[self.n_layers], y)
        
        # do back-prop
        for layer in range(self.n_layers, 0, -1):
            dA = self.layers[layer].Backward(self, layer, dA)
        
        # adjust weight matrix
        self.optimizer.optimize(self)
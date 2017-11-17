import numpy as np

# TODO: apply regularizer on loss functions

class Optimizer:
    
    lr = 0.01
    decay = 0.0

    def __init__(self, lr=0.01, decay=0.0):
        self.lr = lr
        self.decay = decay

    def optimize(self, model):
        return 0

    def lr_decay(self, model):
        self.lr = 1 / (1 + self.decay * model.epoch) * self.lr

class GradientDescent(Optimizer):

    beta = 0.0

    _vdW = [0]
    _vdb = [0]

    def __init__(self, lr=0.01, beta=0.0, decay=0.0):
        self.lr = lr
        self.beta = beta
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._vdW.append(np.zeros(model.W[layer].shape))
            self._vdb.append(np.zeros(model.b[layer].shape))

    def optimize(self, model):
        for layer in range(1, model.n_layers + 1):
            if model.dW[layer] is not None:
                self._vdW[layer] = self.beta * self._vdW[layer] + (1 - self.beta) * model.dW[layer]
                self._vdb[layer] = self.beta * self._vdb[layer] + (1 - self.beta) * model.db[layer]
                model.W[layer] = model.W[layer] - self.lr * self._vdW[layer]
                model.b[layer] = model.b[layer] - self.lr * self._vdb[layer]

class RMSprop(Optimizer):

    beta1 = 0.9
    epsilon = 1e-08

    _sdW = [0]
    _sdb = [0]

    def __init__(self, lr=0.001, beta1=0.9, epsilon=1e-08, decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.epsilon = epsilon
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._sdW.append(np.zeros(model.W[layer].shape))
            self._sdb.append(np.zeros(model.b[layer].shape))

    def optimize(self, model):
        for layer in range(1, model.n_layers + 1):
            if model.dW[layer] is not None:
                self._sdW[layer] =  self.beta1 * self._sdW[layer] + (1 - self.beta1) * np.square(model.dW[layer])
                self._sdb[layer] =  self.beta1 * self._sdb[layer] + (1 - self.beta1) * np.square(model.db[layer])
                model.W[layer] = model.W[layer] - self.lr * model.dW[layer] / (np.sqrt(self._sdW[layer]) + self.epsilon)
                model.b[layer] = model.b[layer] - self.lr * model.db[layer] / (np.sqrt(self._sdb[layer]) + self.epsilon)

class Adam(Optimizer):

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08

    _t = 0

    _vdW = [0]
    _vdb = [0]
    _sdW = [0]
    _sdb = [0] 

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._t = 0
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._vdW.append(np.zeros_like(model.W[layer]))
            self._vdb.append(np.zeros_like(model.b[layer]))
            self._sdW.append(np.zeros_like(model.W[layer]))
            self._sdb.append(np.zeros_like(model.b[layer]))            
 
    def optimize(self, model):
        self._t += 1 # Adam counter
        for layer in range(1, model.n_layers + 1):
            if model.dW[layer] is not None:
                self._vdW[layer] = self.beta_1 * self._vdW[layer] + (1 - self.beta_1) * model.dW[layer]
                self._vdb[layer] = self.beta_1 * self._vdb[layer] + (1 - self.beta_1) * model.db[layer]
                self._sdW[layer] = self.beta_2 * self._sdW[layer] + (1 - self.beta_2) * np.square(model.dW[layer])
                self._sdb[layer] = self.beta_2 * self._sdb[layer] + (1 - self.beta_2) * np.square(model.db[layer])

                _vdW_corr = self._vdW[layer] / (1.0 - np.power(self.beta_1, self._t))
                _vdb_corr = self._vdb[layer] / (1.0 - np.power(self.beta_1, self._t))
                _sdW_corr = self._sdW[layer] / (1.0 - np.power(self.beta_2, self._t))
                _sdb_corr = self._sdb[layer] / (1.0 - np.power(self.beta_2, self._t))

                model.W[layer] = model.W[layer] - self.lr * _vdW_corr / np.sqrt(_sdW_corr + self.epsilon)
                model.b[layer] = model.b[layer] - self.lr * _vdb_corr / np.sqrt(_sdb_corr + self.epsilon)
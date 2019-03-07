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

    _vdweights = [0]
    _vdbias = [0]

    def __init__(self, lr=0.01, beta=0.0, decay=0.0):
        self.lr = lr
        self.beta = beta
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._vdweights.append(np.zeros_like(model.layers[layer].weights))
            self._vdbias.append(np.zeros_like(model.layers[layer].bias))

    def optimize(self, model):
        for layer in range(1, model.n_layers + 1):
            if model.layers[layer].dweights is not None:
                self._vdweights[layer] = self.beta * self._vdweights[layer] + \
                                         (1 - self.beta) * model.layers[layer].dweights
                self._vdbias[layer] = self.beta * self._vdbias[layer] + \
                                      (1 - self.beta) * model.layers[layer].dbias
                model.layers[layer].weights = model.layers[layer].weights - \
                                              self.lr * self._vdweights[layer]
                model.layers[layer].bias = model.layers[layer].bias - \
                                           self.lr * self._vdbias[layer]


class RMSprop(Optimizer):

    beta1 = 0.9
    epsilon = 1e-08

    _sdweights = [0]
    _sdbias = [0]

    def __init__(self, lr=0.001, beta1=0.9, epsilon=1e-08, decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.epsilon = epsilon
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._sdweights.append(np.zeros_like(model.layers[layer].weights))
            self._sdbias.append(np.zeros_like(model.layers[layer].bias))

    def optimize(self, model):
        for layer in range(1, model.n_layers + 1):
            if model.layers[layer].dweights is not None:
                self._sdweights[layer] = self.beta1 * self._sdweights[layer] + \
                                         (1 - self.beta1) * np.square(model.layers[layer].dweights)
                self._sdbias[layer] = self.beta1 * self._sdbias[layer] + \
                                      (1 - self.beta1) * np.square(model.layers[layer].dbias)
                model.layers[layer].weights = model.layers[layer].weights - self.lr * \
                                              model.layers[layer].dweights / \
                                              (np.sqrt(self._sdweights[layer]) + self.epsilon)
                model.layers[layer].bias = model.layers[layer].bias - self.lr * \
                                           model.layers[layer].dbias / \
                                           (np.sqrt(self._sdbias[layer]) + self.epsilon)


class Adam(Optimizer):

    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-08

    _t = 0

    _vdweights = [0]
    _vdbias = [0]
    _sdweights = [0]
    _sdbias = [0]

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self._t = 0
        self.decay = decay

    def init(self, model):
        for layer in range(1, model.n_layers + 1):
            self._vdweights.append(np.zeros_like(model.layers[layer].weights))
            self._vdbias.append(np.zeros_like(model.layers[layer].bias))
            self._sdweights.append(np.zeros_like(model.layers[layer].weights))
            self._sdbias.append(np.zeros_like(model.layers[layer].bias))

    def optimize(self, model):
        self._t += 1  # Adam counter
        for layer in range(1, model.n_layers + 1):
            if model.layers[layer].dweights is not None:
                self._vdweights[layer] = self.beta_1 * self._vdweights[layer] + \
                                         (1 - self.beta_1) * model.layers[layer].dweights
                self._vdbias[layer] = self.beta_1 * self._vdbias[layer] + \
                                      (1 - self.beta_1) * model.layers[layer].dbias
                self._sdweights[layer] = self.beta_2 * self._sdweights[layer] + \
                                         (1 - self.beta_2) * np.square(model.layers[layer].dweights)
                self._sdbias[layer] = self.beta_2 * self._sdbias[layer] + \
                                      (1 - self.beta_2) * np.square(model.layers[layer].dbias)

                _vdweights_corr = self._vdweights[layer] / \
                                  (1.0 - np.power(self.beta_1, self._t))
                _vdbias_corr = self._vdbias[layer] / \
                               (1.0 - np.power(self.beta_1, self._t))
                _sdweights_corr = self._sdweights[layer] / \
                                  (1.0 - np.power(self.beta_2, self._t))
                _sdbias_corr = self._sdbias[layer] / \
                               (1.0 - np.power(self.beta_2, self._t))

                model.layers[layer].weights = model.layers[layer].weights - \
                                              self.lr * _vdweights_corr / \
                                              np.sqrt(_sdweights_corr + self.epsilon)
                model.layers[layer].bias = model.layers[layer].bias - \
                                           self.lr * _vdbias_corr / \
                                           np.sqrt(_sdbias_corr + self.epsilon)

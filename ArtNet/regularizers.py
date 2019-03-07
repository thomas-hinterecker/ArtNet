import numpy as np


class Regularizer:
    lambd = 0.01

    def __init__(self, lambd=0.01):
        self.lambd = lambd

    def regularize_loss(self, weights):
        out = 0
        return out

    def regularize_dweights(self, m, weights, dweights, dbias):
        return dweights, dbias


class L2(Regularizer):

    def regularize_loss(self, weights):
        out = np.sum(self.lambd * np.square(weights))
        return out

    def regularize_dweights(self, m, weights, dweights, dbias):
        dweights = dweights + self.lambd / m * weights
        return dweights, dbias

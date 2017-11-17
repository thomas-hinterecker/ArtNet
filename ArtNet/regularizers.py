import numpy as np

class Regularizer:
    
    lambd = 0.01

    def __init__(self, lambd = 0.01):
        self.lambd = lambd

    def regularize(self, m, weights, dweights, dbias):
        return dweights, dbias


class L2(Regularizer):

    def regularize(self, m, weights, dweights, dbias):
        dweights = dweights + self.lambd / m * weights
        return dweights, dbias

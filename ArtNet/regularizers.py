import numpy as np

# TODO: apply regularizer on loss functions

class Regularizer:
    
    lambd = 0.01

    def __init__(self, lambd = 0.01):
        self.lambd = lambd

    def regularize(self, m, W, dW, db):
        return dw, db


class L2(Regularizer):

    def regularize(self, m, W, dW, db):
        dW = dW + self.lambd / m * W
        return dW, db

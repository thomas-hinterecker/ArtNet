import numpy as np
import math

class Loss:
    
    def L(self, y_pred, y_true):
        return y_pred - y_true

    def L_prime(self, y_pred, y_true):
        return 0

class MeanSquaredError(Loss):

    def L(self, y_pred, y_true):
        m = y_true.shape[1]
        return np.squeeze(1 / m * np.sum(np.square(y_pred - y_true), axis=1))

    def L_prime(self, y_pred, y_true):
        m = y_true.shape[1]
        return 2 / m * np.sum(y_pred - y_true, axis=0, keepdims=True)

class BinaryCrossEntropy(Loss):

    def L(self, y_pred, y_true):
        m = y_true.shape[1]
        return (-1.0 / m) * np.sum(np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1 - y_pred)))

    def L_prime(self, y_pred, y_true):
        return - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))
  
class CategoricalCrossEntropy(Loss):

    epsilon = 1e-15

    def L(self, y_pred, y_true):
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        loss = - np.sum(y_true * np.log(y_pred), axis=1)   
        cost = np.mean(loss, axis=-1)
        return cost

    def L_prime(self, y_pred, y_true):
        return y_pred - y_true
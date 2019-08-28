import numpy as np


class Loss:

    def l_(self, y_pred, y_true):
        return 0

    def l_prime(self, y_pred, y_true):
        return y_pred

    def accuracy(self, y_pred, y_true):
        return 0


class MeanSquaredError(Loss):

    def l_(self, y_pred, y_true):
        m = y_true.shape[0]
        return np.squeeze(1 / m * np.sum(np.square(y_pred - y_true), axis=1))

    def l_prime(self, y_pred, y_true):
        m = y_true.shape[0]
        return 2 / m * np.sum(y_pred - y_true, axis=0, keepdims=True)


class BinaryCrossEntropy(Loss):

    def l_(self, y_pred, y_true):
        m = y_true.shape[0]
        return -1.0 / m * np.sum(np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1 - y_pred)))

    def l_prime(self, y_pred, y_true):
        return - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))


class CategoricalCrossEntropy(Loss):

    def l_(self, y_pred, y_true):
        loss = - np.sum(y_true * np.log(y_pred), axis=1)
        cost = np.mean(loss, axis=-1)
        return cost

    def l_prime(self, y_pred, y_true):
        return y_pred - y_true

    def accuracy(self, y_pred, y_true):
        preds_correct_boolean = np.argmax(
            y_pred.T, axis=1) == np.argmax(y_true.T, axis=1)
        correct_predictions = np.sum(preds_correct_boolean)
        accuracy = correct_predictions / y_pred.T.shape[0]
        return accuracy

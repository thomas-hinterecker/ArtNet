import numpy as np

class ActivationFunction:
    
    def g(self, z):
        return z

    def g_prime(self, z):
        return z

class Linear(ActivationFunction):

    def g(self, z):
        return z

    def g_prime(self, z):
        return 1.0

class Sigmoid(ActivationFunction):

    def g(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def g_prime(self, z):
        o = self.g(z)
        return np.multiply(o, (1.0 - o))

class Tanh(ActivationFunction):

    def g(self, z):
        return np.tanh(z)

    def g_prime(self, z):
        return 1.0 - np.square(self.g(z))

class ReLU(ActivationFunction):

    def g(self, z):
        return np.maximum(z, 0, z)

    def g_prime(self, z):
        return (z > 0) * 1 + (z <= 0) * 0


class Leaky_ReLU(ActivationFunction):

    def g(self, z):
        return np.maximum(0.01 * z, z)

    def g_prime(self, z):
        return (z > 0) * 1 + (z <= 0) * 0.01

class Softmax(ActivationFunction):

    def g(self, z):
        t = np.exp(z - np.max(z, axis=0))
        return t / np.sum(t, axis=0, keepdims=True)

    def g_prime(self, z):
        return 1.0
import numpy as np

# TODO: apply regularizer on loss functions

class Initializer:
    
    def __init__(self):
        pass

    def initialize(self, n_inputs, n_nodes):
        pass 


class Zeros(Initializer):

    def initialize(self, n_inputs, n_nodes):
        return np.zeros((n_nodes, n_inputs))

class Ones(Initializer):

    def initialize(self, n_inputs, n_nodes):
        return np.ones((n_nodes, n_inputs))    

class RandomUniform(Initializer):

    seed = None

    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, n_inputs, n_nodes):
        np.random.seed(seed) if self.seed is not None else None
        return np.random.randn(n_nodes, n_inputs)

class GlorotNormal(Initializer):

    seed = None

    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, n_inputs, n_nodes):
        np.random.seed(seed) if self.seed is not None else None
        return np.random.randn(n_nodes, n_inputs) * np.square(2.0 / (n_inputs + n_nodes))

class GlorotUniform(Initializer):

    seed = None

    def __init__(self, seed=None):
        self.seed = seed

    def initialize(self, n_inputs, n_nodes):
        np.random.seed(seed) if self.seed is not None else None
        return np.random.randn(n_nodes, n_inputs) * np.square(6.0 / (n_inputs + n_nodes))
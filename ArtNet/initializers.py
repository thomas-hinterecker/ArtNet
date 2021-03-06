import numpy as np


class Initializer:

    def __init__(self):
        pass

    def initialize(self, shape):
        pass


class Zeros(Initializer):

    def initialize(self, shape):
        super().__init__()
        return np.zeros(shape)


class Ones(Initializer):

    def initialize(self, shape):
        return np.ones(shape)


class RandomUniform(Initializer):
    seed = None

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def initialize(self, shape):
        np.random.seed(self.seed) if self.seed is not None else None
        return np.random.standard_normal(shape)


class GlorotNormal(Initializer):
    seed = None

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def initialize(self, shape):
        np.random.seed(self.seed) if self.seed is not None else None
        return np.random.standard_normal(shape) * np.square(2.0 / (shape[0] + shape[1]))


class GlorotUniform(Initializer):
    seed = None

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def initialize(self, shape):
        np.random.seed(self.seed) if self.seed is not None else None
        return np.random.standard_normal(shape) * np.square(6.0 / (shape[0] + shape[1]))

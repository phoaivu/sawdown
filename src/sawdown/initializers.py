import numpy as np


class FixedInitializer(object):
    def __init__(self, initializer):
        if initializer.ndim != 1:
            raise ValueError('Must be a vector')
        self._initializer = initializer

    def initialize(self, initializer):
        return self._initializer


class UniformInitializer(object):
    def __init__(self, var_dim=1, low=0.0, high=1.0):
        self._var_dim = var_dim
        self._low = low
        self._high = high

    def initialize(self, initializer):
        if initializer is not None and initializer.shape != (self._var_dim, ):
            raise ValueError('Inconsistent initialization configuration')
        return np.random.uniform(self._low, self._high, (self._var_dim, ))

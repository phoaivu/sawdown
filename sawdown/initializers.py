class InitializerMixIn(object):

    def __init__(self):
        self._initializers = []

    def _setup(self, objective, opti_math, **kwargs):
        pass

    def _initialize(self):
        initializer_val = None
        for initializer in self._initializers:
            initializer_val = initializer.initialize(initializer_val)
        return initializer_val

    def fixed_initializer(self, initializer):
        self._initializers.append(FixedInitializer(initializer))
        return self


class FixedInitializer(object):
    def __init__(self, initializer):
        if initializer.ndim != 1:
            raise ValueError('Must be a vector')
        self._initializer = initializer

    def initialize(self, initializer):
        return self._initializer

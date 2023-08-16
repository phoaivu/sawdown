class FixedInitializer(object):
    def __init__(self, initializer):
        if initializer.ndim != 1:
            raise ValueError('Must be a vector')
        self._initializer = initializer

    def initialize(self, initializer):
        return self._initializer

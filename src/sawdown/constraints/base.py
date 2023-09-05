import numpy as np


class ConstraintsBase(object):
    def clone(self):
        raise NotImplementedError()

    def merge(self, other):
        raise NotImplementedError()

    def is_empty(self):
        return False

    def satisfied(self, x, opti_math):
        raise NotImplementedError()

    def initialization_direction(self, x_k, d_k, opti_math, diary):
        raise NotImplementedError()

    def initialization_steplength(self, k, x_k, d_k, max_steplength, config, opti_math, diary):
        raise NotImplementedError()

    def initialize(self, initializer, config, opti_math, diary):
        """

        :param initializer:
        :param config:
        :param opti_math:
        :param diary:
        :return: initializer
        :rtype: common.Solution
        """
        raise NotImplementedError()

    def direction(self, x_k, d_k, opti_math, diary):
        """

        :param x_k:
        :param d_k:
        :param opti_math:
        :param diary:
        :return: modified direction, usually projected onto constraints.
        :rtype: np.array
        """
        raise NotImplementedError()

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        """

        :param k:
        :param x_k:
        :param d_k:
        :param max_steplength:
        :param opti_math:
        :param diary:
        :return: modified steplength
        :rtype: float
        """
        raise NotImplementedError()


class EmptyConstraints(ConstraintsBase):

    def clone(self):
        return EmptyConstraints()

    def merge(self, other):
        # For the sake of completeness.
        return other.clone()

    def is_empty(self):
        return True

    def satisfied(self, x, opti_math):
        return True

    def initialization_direction(self, x_k, d_k, opti_math, diary):
        return d_k

    def initialization_steplength(self, k, x_k, d_k, max_steplength, config, opti_math, diary):
        return max_steplength

    def initialize(self, initializer, config, opti_math, diary):
        return initializer

    def direction(self, x_k, d_k, opti_math, diary):
        return d_k

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        return max_steplength


class LinearConstraints(ConstraintsBase):
    def __init__(self, a, b):
        if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0]:
            raise ValueError('Invalid shapes of constraints: expect `a` is 2-dimensional and `b` is 1-dimensional')
        self._a = a
        self._b = b

        # compute constraint projection matrix
        # I - (1/a^Ta)aa^T
        self._projectors = self._a[:, :, None] * self._a[:, None, :]
        denominator = np.sum(np.square(self._a), axis=1)
        self._projectors = (np.identity(self._a.shape[1])[None, :, :]
                            - (self._projectors / denominator[:, None, None]))

    def var_dim(self):
        return self._a.shape[1]

    def residuals(self, x):
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim != 2 or self.var_dim() != x.shape[0]:
            raise ValueError('Invalid dimensions')
        residual = np.matmul(self._a, x) + self._b[:, None]
        return residual if residual.shape[-1] > 1 else residual.squeeze(axis=-1)

    def clone(self):
        return self.__class__(self._a.copy(), self._b.copy())

    def merge(self, other):
        if self.var_dim() != other.var_dim():
            raise ValueError('Incompatible variable dimension: one is {}, one is {}'.format(
                self.var_dim(), other.var_dim()))

        return self.__class__(np.vstack((self._a, other._a)), np.concatenate((self._b, other._b)))


class BoundedVariable(object):
    """
    Used in `IntegerityConstraints`.
    """
    def __init__(self, index=0, lower_bound=-np.inf, upper_bound=np.inf):
        self.index = index
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def clone(self):
        return BoundedVariable(self.index, self.lower_bound, self.upper_bound)


class FixedVariable(object):
    """
    Used in `FixedValueConstraints`.
    """
    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value

    def clone(self):
        return FixedVariable(self.index, self.value)

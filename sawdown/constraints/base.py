import numpy as np


class ConstraintsBase(object):

    def var_dim(self):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()

    def merge(self, other):
        raise NotImplementedError()

    def satisfied(self, x):
        raise NotImplementedError()

    def is_empty(self):
        return False

    def setup(self, objective, opti_math, **kwargs):
        pass

    def initialize(self, initializer, diary):
        """

        :param initializer:
        :param diary:
        :return: initializer
        :rtype: common.Solution
        """
        raise NotImplementedError()

    def direction(self, x_k, d_k, diary):
        """

        :param x_k:
        :param d_k:
        :param diary:
        :return: modified direction, usually projected onto constraints.
        :rtype: np.array
        """
        raise NotImplementedError()

    def steplength(self, k, x_k, d_k, max_steplength, diary):
        """

        :param k:
        :param x_k:
        :param d_k:
        :param max_steplength:
        :param diary:
        :return: modified steplength
        :rtype: float
        """
        raise NotImplementedError()


class EmptyConstraints(ConstraintsBase):

    def var_dim(self):
        return -1

    def clone(self):
        return EmptyConstraints()

    def merge(self, other):
        # For the sake of completeness.
        return other.clone()

    def satisfied(self, x):
        return True

    def is_empty(self):
        return True

    def initialize(self, initializer, diary):
        return initializer

    def direction(self, x_k, d_k, diary):
        return d_k

    def steplength(self, k, x_k, d_k, max_steplength, diary):
        return max_steplength


class LinearConstraints(ConstraintsBase):
    def __init__(self, a, b):
        if a.ndim != 2 or b.ndim != 1 or a.shape[0] != b.shape[0]:
            raise ValueError('Invalid shapes of constraints: expect `a` is 2-dimensional and `b` is 1-dimensional')
        self._a = a
        self._b = b
        self._opti_math = None

        # compute constraint projection matrix
        # I - (1/a^Ta)aa^T
        self._projectors = self._a[:, :, None] * self._a[:, None, :]
        denominator = np.sum(np.square(self._a), axis=1)
        self._projectors = (np.identity(self._projectors.shape[1])[None, :, :]
                            - (self._projectors / denominator[:, None, None]))

    def setup(self, objective, opti_math, **kwargs):
        self._opti_math = opti_math

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
        c = self.__class__(self._a.copy(), self._b.copy())
        c.setup(None, self._opti_math)
        return c

    def merge(self, other):
        if self.var_dim() != other.var_dim():
            raise ValueError('Incompatible variable dimension: one is {}, one is {}'.format(
                self.var_dim(), other.var_dim()))

        c = self.__class__(np.vstack((self._a, other._a)), np.concatenate((self._b, other._b)))
        c.setup(None, self._opti_math)
        return c


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

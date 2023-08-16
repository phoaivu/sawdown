import numpy as np
from sawdown import errors
from sawdown.constraints import base


class FixedValueConstraints(base.ConstraintsBase):
    def __init__(self, variables=()):
        if len(variables) == 0:
            raise ValueError('Use EmptyConstraints instead')
        for i in range(len(variables) - 1):
            var = variables[i]
            if any([v.value != var.value for v in variables[i+1:] if v.index == var.index]):
                raise ValueError('Inconsistent constraints for variable index {}'.format(var.index))

        self._variables = variables
        self._indices = [v.index for v in self._variables]
        self._values = np.asarray([v.value for v in self._variables], dtype=float)

    # def var_dim(self):
    #     return self._total_dim

    def clone(self):
        return FixedValueConstraints([v.clone() for v in self._variables])

    def merge(self, other):
        assert isinstance(other, FixedValueConstraints)
        return FixedValueConstraints([v.clone() for v in self._variables + other._variables])
        # if self._total_dim != other._total_dim and self._total_dim != -1 and other._total_dim != -1:
        #     raise ValueError('Cannot merge 2 constraints of different variable dimensions')
        # dim = self._total_dim if self._total_dim != -1 else other._total_dim
        # return FixedValueConstraints([v.clone() for v in self._variables + other._variables], dim)

    # def to_equalities(self, var_dim):
    #     """
    #     Returns a LinearEqualityConstraints.
    #     """
    #     var_dim = var_dim if var_dim > 0 else self.var_dim()
    #     if var_dim <= max(self._indices):
    #         raise ValueError('var_dim has to be at least {}, given {}'.format(max(self._indices) + 1, var_dim))
    #     n_constraints = len(self._variables)
    #     a = np.zeros((n_constraints, var_dim), dtype=float)
    #     a[list(range(n_constraints)), self._indices] = 1.
    #     b = -self._values.copy()
    #     return LinearEqualityConstraints(a, b)

    def satisfied(self, x, opti_math):
        return np.all(opti_math.equal_zeros(x[self._indices] - self._values))

    def initialization_direction(self, x_k, d_k, opti_math, diary):
        direction = np.zeros_like(x_k, dtype=float)
        direction[self._indices] = self._values - x_k[self._indices]
        if d_k is not None:
            direction = d_k + direction
        return direction

    def initialization_steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        return np.minimum(max_steplength, 1.)

    def initialize(self, initializer, config, opti_math, diary):
        if initializer is None:
            raise ValueError('Unknown variable dimension. Try hinting the optimizer '
                             'via one of the initialization method')
        if max(self._indices) >= initializer.size:
            raise ValueError('There is fixed value constraint for variable #{}, '
                             'but initialized variable dimension is {}'.format(max(self._indices), initializer.size))

        return opti_math.optimize(initializer, self.satisfied, self.initialization_direction,
                                  self.initialization_steplength, config.initialization_max_iters, diary)

    def direction(self, x_k, d_k, opti_math, diary):
        projected_d_k = d_k.copy()
        projected_d_k[self._indices] = 0.
        return projected_d_k

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        if np.any(opti_math.non_zeros(d_k[self._indices])):
            return 0.
        return max_steplength


class LinearEqualityConstraints(base.LinearConstraints):
    def __init__(self, a, b):
        base.LinearConstraints.__init__(self, a, b)

        # precompute the master projection matrix: product of all projectors
        self._master_projector = self._projectors[0, :, :]
        for i in range(self._projectors.shape[0] - 1):
            self._master_projector = np.matmul(self._projectors[i+1, :, :], self._master_projector)
        if not np.all(np.isfinite(self._master_projector)):
            raise RuntimeError('Machine is tired, or math is wrong')

    def satisfied(self, x, opti_math):
        return np.all(opti_math.equal_zeros(self.residuals(x)))

    def __least_square_objective(self, x_k):
        return 0.5 * np.sum(np.square(np.matmul(self._a, x_k[:, None]) + self._b[:, None]), axis=0).squeeze()

    def initialization_direction(self, x_k, d_k, opti_math, diary):
        # This is the direction as if doing a least-square
        direction = -np.matmul(self._a.T, np.matmul(self._a, x_k[:, None]) + self._b[:, None]).squeeze()
        if d_k is not None:
            direction = d_k + direction
        return direction

    def initialization_steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        # Quadratic interpolation.
        delta = 0.
        beta = np.matmul(-d_k.T, d_k)
        alpha = self.__least_square_objective(x_k + d_k) - self.__least_square_objective(x_k) - beta
        if alpha * beta < 0.:
            delta = min((-beta / (2. * alpha)).squeeze(), 1)
        return np.minimum(max_steplength, delta)

    def initialize(self, initializer, config, opti_math, diary):
        """

        :param initializer:
        :param config:
        :param opti_math:
        :param diary:
        :return: initializer
        :rtype: np.ndarray
        """
        # Doing a least-square, fully aware of np.linalg.lstsq().
        x_0 = np.zeros((self.var_dim(),), dtype=float) if initializer is None else initializer.copy()
        if x_0.shape != (self.var_dim(), ):
            raise errors.InitializationError('Given a mismatch dimension initializer')

        return opti_math.optimize(x_0, self.satisfied, self.initialization_direction,
                                  self.initialization_steplength, config.initialization_max_iters, diary)

    def direction(self, x_k, d_k, opti_math, diary):
        return np.matmul(self._master_projector, d_k[:, None]).squeeze(axis=-1)

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
        # If there are constraints whose gradients are not orthogonal to d_k, then return 0.
        a_d_k = np.matmul(self._a, d_k[:, None]).squeeze(axis=1)
        if np.any(opti_math.non_zeros(a_d_k)):
            return 0.
        return max_steplength

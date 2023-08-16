import numpy as np
from sawdown import errors
from sawdown.constraints import base


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

        def _objective(x_k):
            return 0.5 * np.sum(np.square(np.matmul(self._a, x_k[:, None]) + self._b[:, None]), axis=0).squeeze()

        def _director(x_k, _diary):
            return -np.matmul(self._a.T, np.matmul(self._a, x_k[:, None]) + self._b[:, None]).squeeze()

        def _stepper(k, x_k, d_k, _diary):
            # Quadratic interpolation.
            delta = 0.
            beta = np.matmul(-d_k.T, d_k)
            alpha = _objective(x_k + d_k) - _objective(x_k) - beta
            if alpha * beta < 0.:
                delta = min((-beta / (2. * alpha)).squeeze(), 1)
            return delta

        return opti_math.optimize(x_0, self.satisfied, _director, _stepper, config.initialization_max_iters, diary)

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


class FixedValueConstraints(base.ConstraintsBase):
    def __init__(self, variables, total_dim=-1):
        if len(variables) == 0:
            raise ValueError('Use EmptyConstraints instead')
        if -1 < total_dim <= max(v.index for v in variables):
            raise ValueError('total_dim must be at least greater than the max. variable index')
        for i in range(len(variables) - 1):
            var = variables[i]
            if any([v.value != var.value for v in variables[i+1:] if v.index == var.index]):
                raise ValueError('Inconsistent constraints for variable index {}'.format(var.index))

        self._variables = variables
        self._total_dim = total_dim
        self._indices = [v.index for v in self._variables]
        self._values = [v.value for v in self._variables]

    def var_dim(self):
        return self._total_dim

    def clone(self):
        return FixedValueConstraints([v.clone() for v in self._variables], self._total_dim)

    def merge(self, other):
        assert isinstance(other, FixedValueConstraints)
        if self._total_dim != other._total_dim and self._total_dim != -1 and other._total_dim != -1:
            raise ValueError('Cannot merge 2 constraints of different variable dimensions')
        dim = self._total_dim if self._total_dim != -1 else other._total_dim
        return FixedValueConstraints([v.clone() for v in self._variables + other._variables], dim)

    def satisfied(self, x, opti_math):
        return np.all(opti_math.equal_zeros(x[self._indices] - np.asarray(self._values, dtype=float)))

    def to_equalities(self, var_dim):
        """
        Returns a LinearEqualityConstraints.
        """
        var_dim = var_dim if var_dim > 0 else self.var_dim()
        if var_dim <= max(self._indices):
            raise ValueError('var_dim has to be at least {}, given {}'.format(max(self._indices) + 1, var_dim))
        n_constraints = len(self._variables)
        a = np.zeros((n_constraints, var_dim), dtype=float)
        a[list(range(n_constraints)), self._indices] = 1.
        b = -np.fromiter(self._values, dtype=float)
        return LinearEqualityConstraints(a, b)

    def direction(self, x_k, d_k, opti_math, diary):
        projected_d_k = d_k.copy()
        projected_d_k[self._indices] = 0.
        return projected_d_k

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        if np.any(opti_math.non_zeros(d_k[self._indices])):
            return 0.
        return max_steplength

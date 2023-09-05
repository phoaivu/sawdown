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

    def clone(self):
        return FixedValueConstraints([v.clone() for v in self._variables])

    def merge(self, other):
        assert isinstance(other, FixedValueConstraints)
        return FixedValueConstraints([v.clone() for v in self._variables + other._variables])

    def to_equalities(self, var_dim):
        """
        Returns a LinearEqualityConstraints.
        """
        if var_dim <= max(self._indices):
            raise ValueError('var_dim has to be at least {}, given {}'.format(max(self._indices) + 1, var_dim))
        n_constraints = len(self._variables)
        a = np.zeros((n_constraints, var_dim), dtype=float)
        a[list(range(n_constraints)), self._indices] = 1.
        b = -self._values.copy()
        return LinearEqualityConstraints(a, b)


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

    def __least_square_grad(self, x_k):
        return np.matmul(self._a.T, np.matmul(self._a, x_k[:, None]) + self._b[:, None]).squeeze()

    def initialization_direction(self, x_k, d_k, opti_math, diary):
        return -self.__least_square_grad(x_k)

    def initialization_steplength(self, k, x_k, d_k, max_steplength, config, opti_math, diary):
        # Quadratic interpolation.
        delta = 0.
        beta = np.matmul(self.__least_square_grad(x_k).T, d_k)
        g_max = self.__least_square_objective(x_k + max_steplength * d_k)
        alpha = (g_max - (beta * max_steplength) - self.__least_square_objective(x_k)) / np.square(max_steplength)
        if alpha * beta < 0.:
            delta = min((-beta / (2. * alpha)).squeeze(), max_steplength)
        return delta

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
                                  self.initialization_steplength, config, diary)

    def direction(self, x_k, d_k, opti_math, diary):
        # projected_d_k = np.matmul(self._master_projector, d_k[:, None]).squeeze(axis=-1)
        # if not np.all(opti_math.equal_zeros(np.matmul(self._a, projected_d_k[:, None]).squeeze(axis=-1))):
        projected_d_k = d_k.copy()[:, None]
        for i in range(self._projectors.shape[0]):
            projected_d_k = np.matmul(self._projectors[i, :, :], projected_d_k)
            dot = np.matmul(self._a[i:(i+1), :], projected_d_k).squeeze(axis=-1)
            if not np.all(opti_math.equal_zeros(dot)):
                print(dot)
                print(opti_math.equal_zeros(dot))
                print(self._projectors[i, :, :].tolist())
                print(self._a[i, :].tolist())
                assert False
        return projected_d_k.squeeze(axis=-1)

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
            diary.set_items(equality_projected_d_k=a_d_k.copy(), msg_equalities='Zero out the steplength')
            return 0.
        return max_steplength

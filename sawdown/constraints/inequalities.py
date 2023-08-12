import numpy as np
from sawdown import errors
from sawdown.constraints import base


class InequalityConstraints(object):

    def initialization_direction(self, x_k, diary):
        raise NotImplementedError()

    def initialization_steplength(self, k, x_k, d_k, diary):
        raise NotImplementedError()


class EmptyInequalityConstraints(base.EmptyConstraints, InequalityConstraints):
    def clone(self):
        return EmptyInequalityConstraints()


class LinearInequalityConstraints(base.LinearConstraints, InequalityConstraints):
    def __init__(self, a, b):
        base.LinearConstraints.__init__(self, a, b)
        self._lagrange_bouncing = False
        self._active_constraints = None

    def initialization_direction(self, x_k, diary):
        residuals = self.residuals(x_k)
        mask = self._opti_math.true_negative(residuals)
        lengths = -residuals[mask] / np.sum(np.square(self._a[mask, :]), axis=1)
        return np.matmul(lengths[None, :], self._a[mask, :]).squeeze(axis=0)

    def initialization_steplength(self, k, x_k, d_k, diary):
        return 1. + np.exp(-float(k) / self._opti_math.initialization_decay_steps)

    def satisfied(self, x):
        return np.all(self._opti_math.non_negative(self.residuals(x)))
        # residuals = self.residuals(x)
        # return np.all(residuals >= -np.sqrt(common.EPSILON))

    def initialize(self, initializer, diary):
        """

        :param initializer:
        :param diary:
        :return: initializer
        :rtype: sawdown.diaries.Solution
        """
        x_0 = np.zeros((self.var_dim(),), dtype=float) if initializer is None else initializer.copy()
        if x_0.shape != (self.var_dim(), ):
            raise errors.InitializationError('Given a mismatch dimension initializer')

        return self._opti_math.optimize(x_0, self.satisfied, self.initialization_direction,
                                        self.initialization_steplength, diary)

    def direction(self, x_k, d_k, diary):
        # Probably not needed.
        self._active_constraints = self._opti_math.equal_zeros(self.residuals(x_k))

        if self._lagrange_bouncing:
            d_k += np.matmul(1. * self._active_constraints[None, :], self._a).T.squeeze(axis=1)

        # Only project d_k onto the constraints that go against d_k
        # i.e. those i's that makes cosine(A[i, :], d_k) <= 0
        projected_constraints = np.logical_xor(self._active_constraints, self._active_constraints)
        projected_constraints[self._active_constraints] = self._opti_math.non_positive(
            np.matmul(self._a[self._active_constraints, :], d_k))

        if np.any(projected_constraints):
            # Project d_k
            active_projectors = self._projectors[projected_constraints, :, :]  # num_active x num_vars x num_vars
            # num_active x num_vars
            projected_d_k = np.matmul(active_projectors, d_k[None, :, None]).squeeze(axis=-1)

            # pick the constraint whose projection of d_k is smallest, i.e. the innermost constraint
            d_projected_idx = np.argmin(np.sum(np.square(projected_d_k), axis=1), axis=0)
            constraint_idx = np.nonzero(projected_constraints)[0][d_projected_idx].squeeze()

            # turn off all other constraints, except constrain_idx, which is the one being projected on.
            self._active_constraints[projected_constraints] = False
            self._active_constraints[constraint_idx] = True

            diary.set_items(constraint_idx=constraint_idx, msg='Projected d_k onto constraint')

            d_k = projected_d_k[d_projected_idx, :].copy()
        else:
            # no more constraints
            self._active_constraints = np.logical_xor(self._active_constraints, self._active_constraints)
        return d_k

    def steplength(self, k, x_k, d_k, max_steplength, diary):
        residual_before = self.residuals(x_k)
        assert np.all(self._opti_math.non_negative(residual_before)), 'On the wrong side of the universe.'

        # The length needed to travel along d_k and arrives right on the constraint curves.
        # A(x + delta * d_k) + b = (Ax + b) + delta*(A*d_k) = 0
        # deltas = -(Ax + b) ./ A*d_k
        a_d_k = np.matmul(self._a, d_k[:, None]).squeeze(axis=1)
        # Only consider constraints that has its gradients (i.e. constraint_a[i]) opposite in direction with d_k
        # mask = a_d_k < -np.sqrt(common.EPSILON)
        mask = self._opti_math.negative(a_d_k)
        if np.any(mask):
            deltas = np.ones_like(a_d_k, dtype=float) * max_steplength
            # deltas[mask] = -residual_before[mask] / a_d_k[mask]
            deltas[mask] = (np.minimum(0., -residual_before[mask])) / a_d_k[mask]
            delta = np.min(deltas)
            if delta < max_steplength:
                new_active_constraints = self._opti_math.equals(deltas, delta)
                self._active_constraints = np.logical_or(self._active_constraints, new_active_constraints)
                max_steplength = delta

        while (np.any(self._opti_math.negative(self.residuals(x_k + max_steplength * d_k)))
                and self._opti_math.true_negative(-max_steplength)):
            max_steplength = np.maximum(0., max_steplength * (1. - self._opti_math.epsilon))

        return max_steplength

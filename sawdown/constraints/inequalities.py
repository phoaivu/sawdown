import numpy as np
from sawdown import errors
from sawdown.constraints import base


class InequalityConstraints(object):

    def initialization_direction(self, x_k, opti_math, diary):
        raise NotImplementedError()

    def initialization_steplength(self, k, x_k, d_k, opti_math, diary):
        raise NotImplementedError()


class EmptyInequalityConstraints(base.EmptyConstraints, InequalityConstraints):
    def clone(self):
        return EmptyInequalityConstraints()


class BoundConstraints(base.ConstraintsBase, InequalityConstraints):
    def __init__(self, variables=()):
        variables = [v for v in variables if np.isfinite(v.lower_bound) or np.isfinite(v.upper_bound)]
        if any(v.lower_bound >= v.upper_bound for v in variables):
            raise ValueError('Invalid lower bound and upper bound for integer variables')
        if len(set(v.index for v in variables)) < len(variables):
            raise ValueError('Duplicated entries in the list of bound variables')
        if len(variables) == 0:
            raise ValueError('Empty bound constraints')
        self._variables = variables
        self._indices = [v.index for v in self._variables]
        self._lower_bounds = np.asarray([v.lower_bound for v in self._variables], dtype=float)
        self._upper_bounds = np.asarray([v.upper_bound for v in self._variables], dtype=float)

    # TODO: implement this
    def satisfied(self, x, opti_math):
        return opti_math.in_bounds(x[self._indices], self._lower_bounds, self._upper_bounds)

    def direction(self, x_k, d_k, opti_math, diary):
        if not np.all(opti_math.lt(self._lower_bounds, self._upper_bounds)):
            raise ValueError('Tiny bound constraint on some variables. Use fixed value constraints instead')
        active_bounds = np.logical_or(opti_math.equals(x_k[self._indices], self._lower_bounds),
                                      opti_math.equals(x_k[self._indices], self._upper_bounds))
        through_lower = np.logical_and(opti_math.non_positive(d_k[self._indices]),
                                       np.isfinite(self._lower_bounds))
        through_upper = np.logical_and(opti_math.non_negative(d_k[self._indices]),
                                       np.isfinite(self._upper_bounds))
        projections = np.logical_and(active_bounds, np.logical_or(through_lower, through_upper))
        if np.any(projections):
            constraint_idx = np.argmin(np.square(d_k[projections]))
            constraint_idx = np.nonzero(projections)[0][constraint_idx].squeeze()
            d_k[self._indices[constraint_idx]] = 0.
            diary.set_items(bound_constraint_idx=self._indices[constraint_idx],
                            msg_bound_constraint='Projected d_k onto bound constraint')
        return d_k

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        assert self.satisfied(x_k, opti_math)
        through_lower = np.logical_and(opti_math.negative(d_k[self._indices]),
                                       np.isfinite(self._lower_bounds))
        through_upper = np.logical_and(opti_math.positive(d_k[self._indices]),
                                       np.isfinite(self._upper_bounds))
        if np.any(np.logical_or(through_lower, through_upper)):
            deltas = np.ones_like(self._lower_bounds, dtype=float) * max_steplength
            deltas[through_lower] = (self._lower_bounds[through_lower] - x_k[through_lower]) / d_k[through_lower]
            deltas[through_upper] = (self._upper_bounds[through_upper] - x_k[through_upper]) / d_k[through_upper]
            max_steplength = np.min(deltas)
        while (not self.satisfied(x_k + max_steplength * d_k, opti_math)) and opti_math.true_negative(-max_steplength):
            max_steplength = np.maximum(0., max_steplength * (1. - opti_math.epsilon))
        return max_steplength


class LinearInequalityConstraints(base.LinearConstraints, InequalityConstraints):
    def __init__(self, a, b):
        base.LinearConstraints.__init__(self, a, b)
        self._lagrange_bouncing = False
        self._active_constraints = None

    def initialization_direction(self, x_k, opti_math, diary):
        residuals = self.residuals(x_k)
        mask = opti_math.true_negative(residuals)
        lengths = -residuals[mask] / np.sum(np.square(self._a[mask, :]), axis=1)
        return np.matmul(lengths[None, :], self._a[mask, :]).squeeze(axis=0)

    def initialization_steplength(self, k, x_k, d_k, opti_math, diary):
        return 1. + np.exp(-float(k) / opti_math.initialization_decay_steps)

    def satisfied(self, x, opti_math):
        return np.all(opti_math.non_negative(self.residuals(x)))

    def initialize(self, initializer, config, opti_math, diary):
        x_0 = np.zeros((self.var_dim(),), dtype=float) if initializer is None else initializer.copy()
        if x_0.shape != (self.var_dim(), ):
            raise errors.InitializationError('Given a mismatch dimension initializer')

        return opti_math.optimize(x_0, self.satisfied, self.initialization_direction,
                                  self.initialization_steplength, config.initialization_max_iters, diary)

    def direction(self, x_k, d_k, opti_math, diary):
        # Probably not needed.
        self._active_constraints = opti_math.equal_zeros(self.residuals(x_k))

        if self._lagrange_bouncing:
            d_k += np.matmul(1. * self._active_constraints[None, :], self._a).T.squeeze(axis=1)

        # Only project d_k onto the constraints that go against d_k
        # i.e. those i's that makes cosine(A[i, :], d_k) <= 0
        projected_constraints = np.logical_xor(self._active_constraints, self._active_constraints)
        projected_constraints[self._active_constraints] = opti_math.non_positive(
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

    def steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
        residual_before = self.residuals(x_k)
        assert np.all(opti_math.non_negative(residual_before)), 'On the wrong side of the universe.'

        # The length needed to travel along d_k and arrives right on the constraint curves.
        # A(x + delta * d_k) + b = (Ax + b) + delta*(A*d_k) = 0
        # deltas = -(Ax + b) ./ A*d_k
        a_d_k = np.matmul(self._a, d_k[:, None]).squeeze(axis=1)
        # Only consider constraints that has its gradients (i.e. constraint_a[i]) opposite in direction with d_k
        # mask = a_d_k < -np.sqrt(common.EPSILON)
        mask = opti_math.negative(a_d_k)
        if np.any(mask):
            deltas = np.ones_like(a_d_k, dtype=float) * max_steplength
            # deltas[mask] = -residual_before[mask] / a_d_k[mask]
            deltas[mask] = (np.minimum(0., -residual_before[mask])) / a_d_k[mask]
            delta = np.min(deltas)
            if delta < max_steplength:
                new_active_constraints = opti_math.equals(deltas, delta)
                self._active_constraints = np.logical_or(self._active_constraints, new_active_constraints)
                max_steplength = delta

        while (np.any(opti_math.negative(self.residuals(x_k + max_steplength * d_k)))
                and opti_math.true_negative(-max_steplength)):
            max_steplength = np.maximum(0., max_steplength * (1. - opti_math.epsilon))

        return max_steplength

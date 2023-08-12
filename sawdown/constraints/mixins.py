import numpy as np

from sawdown import errors
from sawdown.constraints import base, inequalities, equalities, integerities


class ConstraintMerger(object):

    def _merge_constraints(self, first, second):
        if first.__class__ == second.__class__:
            return first.merge(second)

        _mergers = [
            (base.EmptyConstraints, base.ConstraintsBase,
             lambda f, s: s.clone()),
            (equalities.LinearEqualityConstraints, equalities.FixedValueConstraints,
             lambda f, s: f.merge(s.to_equalities(f.var_dim()))),
            (inequalities.LinearInequalityConstraints, integerities.IntegerityConstraints,
             lambda f, s: f.merge(s.range_constraints(f.var_dim())))
        ]

        for (this, that, merger) in _mergers:
            if isinstance(first, this) and isinstance(second, that):
                return merger(first, second)
            if isinstance(second, this) and isinstance(first, that):
                return merger(second, first)

        raise errors.IncompatibleConstraintsError(first.__class__.__name__, second.__class__.__name__)


class ConstraintsMixIn(ConstraintMerger):

    def __init__(self):
        ConstraintMerger.__init__(self)
        self._inequality_constraints = inequalities.EmptyInequalityConstraints()
        self._equality_constraints = base.EmptyConstraints()
        self._fixed_value_constraints = base.EmptyConstraints()
        self.__constraints_opti_math = None

    def linear_inequality_constraints(self, a, b):
        """
        ax + b >= 0
        :param a:
        :param b:
        :return:
        """
        self._inequality_constraints = self._merge_constraints(self._inequality_constraints,
                                                               inequalities.LinearInequalityConstraints(a, b))
        return self

    def linear_equality_constraints(self, a, b):
        """
        ax + b = 0
        :param a:
        :param b:
        :return:
        """
        self._equality_constraints = self._merge_constraints(self._equality_constraints,
                                                             equalities.LinearEqualityConstraints(a, b))
        return self

    def fixed_value_constraint(self, var_index=0, value=0., total_dim=-1):
        """
        The given variable take pre-described value.
        End-users normally do not use this (because you don't need to optimize a fixed variable!),
        it is used mainly to speed-up binary and integer programs.
        """
        self._fixed_value_constraints = self._merge_constraints(
            self._fixed_value_constraints,
            equalities.FixedValueConstraints([base.FixedVariable(var_index, value)], total_dim))
        return self

    """
    Called by the optimizer
    """

    def _setup(self, objective, opti_math, **kwargs):
        for c in [self._inequality_constraints, self._equality_constraints, self._fixed_value_constraints]:
            c.setup(objective, opti_math, **kwargs)
        self.__constraints_opti_math = opti_math

    def append_constraints(self, inequality_constraints, equality_constraints, fixed_values):
        if inequality_constraints is not None:
            self._inequality_constraints = self._merge_constraints(self._inequality_constraints, inequality_constraints)
        if equality_constraints is not None:
            self._equality_constraints = self._merge_constraints(self._equality_constraints, equality_constraints)
        if fixed_values is not None:
            self._fixed_value_constraints = self._merge_constraints(self._fixed_value_constraints, fixed_values)
        return self

    def _prestep(self, k, x_k, diary):
        """

        :param k:
        :param x_k:
        :param diary:
        :return: modified x_k, if needed.
        :rtype: np.array
        """
        if not self._fixed_value_constraints.satisfied(x_k):
            raise RuntimeError('Fixed-value constraints are violated')
        if not self._equality_constraints.satisfied(x_k):
            raise RuntimeError('Equality constraints are violated.')
        if not self._inequality_constraints.satisfied(x_k):
            # tuck it back, which is kinda over-protective.
            try:
                with diary.sub() as sub_diary:
                    sub_diary.set_items(msg='Tuck it back')
                    return self._initialize(x_k, sub_diary)
            except errors.InitializationError as ex:
                raise RuntimeError('Failed to tuck it back: {}'.format(ex.reason))
        return x_k

    def _initialize(self, initializer, diary):
        """

        :param initializer:
        :param diary:
        :return: initializer
        :rtype: np.ndarray
        """
        if self._equality_constraints.is_empty() and self._fixed_value_constraints.is_empty():
            return self._inequality_constraints.initialize(initializer, diary)

        # This is only done in initialization.
        # During optimization, the proper (hopefully more economical) way is used.
        merged_equality_constraints = self._merge_constraints(self._equality_constraints, self._fixed_value_constraints)
        # merged_equality_constraints.setup(objective=None, opti_math=self.__constraints_opti_math)
        x_0 = merged_equality_constraints.initialize(initializer, diary)
        if self._inequality_constraints.is_empty():
            return x_0

        diary.set_items(x=x_0.copy(), msg='Initialized for equalities constraints. Now for both.')

        def _satisfied(x_k):
            return merged_equality_constraints.satisfied(x_k) and self._inequality_constraints.satisfied(x_k)

        def _director(x_k, _diary):
            d_k = self._inequality_constraints.initialization_direction(x_k, _diary)
            diary.set_items(inequality_direction=d_k.copy())
            return merged_equality_constraints.direction(x_k, d_k, _diary)

        def _stepper(k, x_k, d_k, _diary):
            length = self._inequality_constraints.initialization_steplength(k, x_k, d_k, _diary)
            return merged_equality_constraints.steplength(k, x_k, d_k, length, _diary)

        return self.__constraints_opti_math.optimize(x_0, _satisfied, _director, _stepper, diary)

    def _direction(self, k, x_k, d_k, diary):
        """

        :param x_k:
        :param d_k:
        :param diary:
        :return: modified direction, usually projected onto constraints.
        :rtype: np.array
        """
        # Project d_k sequentially to the innermost active inequality constraint, then all equality constraints
        for constraints in [self._inequality_constraints, self._equality_constraints, self._fixed_value_constraints]:
            d_k = constraints.direction(x_k, d_k, diary)
        return d_k

    def _steplength(self, k, x_k, d_k, max_steplength, diary):
        """

        :param k:
        :param x_k:
        :param d_k:
        :param max_steplength:
        :param diary:
        :return: modified steplength
        :rtype: float
        """
        delta = max_steplength
        for constraints in [self._inequality_constraints, self._equality_constraints, self._fixed_value_constraints]:
            delta = constraints.steplength(k, x_k, d_k, delta, diary)
        return delta


class IntegerConstraintsMixIn(ConstraintMerger):
    """
    """

    def __init__(self):
        ConstraintMerger.__init__(self)
        self._integer_constraints = integerities.EmptyIntegerityConstraints()

    def integer_constraints(self, var_indices=(), lower_bounds=(), upper_bounds=(), total_dim=-1):
        for idx, l, u in zip(var_indices, lower_bounds, upper_bounds):
            self.integer_constraint(idx, l, u, total_dim)
        return self

    def integer_constraint(self, var_index=0, lower_bound=-np.inf, upper_bound=np.inf, total_dim=-1):
        """
        Users don't need to explicitly specify bound constraints for integer variables.
        It will be added automatically.
        :param var_index:
        :param lower_bound:
        :param upper_bound:
        :param total_dim:
        :return:
        """
        if not lower_bound < upper_bound:
            raise ValueError('Impossible bounds for variable index {}: [{}, {}]'.format(
                var_index, lower_bound, upper_bound))
        self._integer_constraints = self._merge_constraints(
            self._integer_constraints,
            integerities.IntegerityConstraints([base.BoundedVariable(var_index, lower_bound, upper_bound)], total_dim))
        return self

    def binary_constraints(self, var_indices=(), total_dim=-1):
        for idx in var_indices:
            self.integer_constraint(idx, 0., 1., total_dim)
        return self

    def binary_constraint(self, var_index, total_dim=-1):
        return self.integer_constraint(var_index, 0., 1., total_dim)

    """
    Called by the MIP Solver
    """

    def _setup(self, objective, opti_math, **kwargs):
        self._integer_constraints.setup(objective, opti_math, **kwargs)

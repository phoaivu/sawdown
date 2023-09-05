import numpy as np

from sawdown import errors
from sawdown.constraints import base, inequalities, equalities
from sawdown.proto import serializer


def merge_constraints(first, second):
    if first.__class__ == second.__class__:
        return first.merge(second)

    _mergers = [
        (base.EmptyConstraints, base.ConstraintsBase,
         lambda f, s: s.clone()),
        # (equalities.LinearEqualityConstraints, equalities.FixedValueConstraints,
        #  lambda f, s: f.merge(s.to_equalities(f.var_dim()))),
        # (inequalities.LinearInequalityConstraints, integerities.IntegerityConstraints,
        #  lambda f, s: f.merge(s.range_constraints(f.var_dim())))
    ]

    for (this, that, merger) in _mergers:
        if isinstance(first, this) and isinstance(second, that):
            return merger(first, second)
        if isinstance(second, this) and isinstance(first, that):
            return merger(second, first)

    raise errors.IncompatibleConstraintsError(first.__class__.__name__, second.__class__.__name__)


class ConstraintsMixIn(object):

    def __init__(self, proto_problem):
        # NOTE: the missing type of constraints are non-linear inequalities/equalities,
        # which, when supported, implemented via a merge into self._inequality_constraints
        # and self._equality_constraints.
        self._inequality_constraints = base.EmptyConstraints()
        self._equality_constraints = base.EmptyConstraints()
        self._bound_constraints = base.EmptyConstraints()
        self._fixed_value_constraints = base.EmptyConstraints()

        for msg in proto_problem.linear_inequalities:
            self._inequality_constraints = self._inequality_constraints.merge(
                inequalities.LinearInequalityConstraints(
                    a=serializer.decode_ndarray(msg.a), b=serializer.decode_ndarray(msg.b)))
        for msg in proto_problem.linear_equalities:
            self._equality_constraints = self._equality_constraints.merge(
                equalities.LinearEqualityConstraints(
                    a=serializer.decode_ndarray(msg.a), b=serializer.decode_ndarray(msg.b)))

        if len(proto_problem.fixed_value_constraints) > 0:
            self._fixed_value_constraints = equalities.FixedValueConstraints(
                [base.FixedVariable(index=v.var_index, value=v.value) for v in proto_problem.fixed_value_constraints])
        if len(proto_problem.bound_constraints) > 0:
            self._bound_constraints = inequalities.BoundConstraints(
                [base.BoundedVariable(index=v.var_index, lower_bound=v.lower, upper_bound=v.upper)
                 for v in proto_problem.bound_constraints])

        # This will be filled in when initialization started.
        self._working_equality_constraints = None

    """
    Called by the optimizer
    """

    def _prestep(self, k, x_k, opti_math, diary):
        """

        :param k:
        :param x_k:
        :param opti_math:
        :param diary:
        :return: modified x_k, if needed.
        :rtype: np.array
        """
        assert self._working_equality_constraints is not None
        if not self._working_equality_constraints.satisfied(x_k, opti_math):
            raise RuntimeError('Equality and/or Fixed-value constraints are violated')
        if not self._inequality_constraints.satisfied(x_k, opti_math):
            raise RuntimeError('Inequality constraints are violated')
        if not self._bound_constraints.satisfied(x_k, opti_math):
            raise RuntimeError('Bound constraints are violated')
        return x_k

    def __equality_initialize(self, initializer, config, opti_math, diary):
        """
        Initialize with equality and fixed-vale constraints.
        """
        if self._fixed_value_constraints.is_empty():
            self._working_equality_constraints = self._equality_constraints
        else:
            var_dim = 0
            if initializer is not None:
                var_dim = initializer.shape[0]
            elif hasattr(self._equality_constraints, 'var_dim'):
                var_dim = self._equality_constraints.var_dim()
            if var_dim == 0:
                raise RuntimeError('Unknown variable dimension. Try hinting the optimizer '
                                   'via the initialization methods')
            fixed_equalities = self._fixed_value_constraints.to_equalities(var_dim=var_dim)
            self._working_equality_constraints = merge_constraints(self._equality_constraints, fixed_equalities)

        return self._working_equality_constraints.initialize(initializer, config, opti_math, diary)

    def __inequality_initialize(self, initializer, config, opti_math, diary):
        if self._inequality_constraints.is_empty() and self._bound_constraints.is_empty():
            return initializer

        empty_equalities = self._working_equality_constraints.is_empty()
        if empty_equalities and self._inequality_constraints.is_empty():
            return self._bound_constraints.initialize(initializer, config, opti_math, diary)
        if empty_equalities and self._bound_constraints.is_empty():
            return self._inequality_constraints.initialize(initializer, config, opti_math, diary)

        diary.set_items(x=None if initializer is None else initializer.copy(),
                        msg_constrained_initialize='Initialized for equalities constraints. Now for both.')

        def _satisfied(x_k, _opti_math):
            return all(map(lambda c: c.satisfied(x_k, _opti_math),
                           [self._working_equality_constraints, self._inequality_constraints, self._bound_constraints]))

        def _director(x_k, d_k, _opti_math, _diary):
            for c in [self._inequality_constraints, self._bound_constraints]:
                d_k = c.initialization_direction(x_k, d_k, _opti_math, _diary)
            diary.set_items(inequality_direction=d_k.copy())
            d_k = self._working_equality_constraints.direction(x_k, d_k, _opti_math, _diary)
            return d_k

        def _stepper(k, x_k, d_k, length, _config, _opti_math, _diary):
            for c in [self._inequality_constraints, self._bound_constraints]:
                length = c.initialization_steplength(k, x_k, d_k, length, _config, _opti_math, _diary)
            length = self._working_equality_constraints.steplength(k, x_k, d_k, length, _opti_math, _diary)
            return length

        if initializer is None:
            if hasattr(self._inequality_constraints, 'var_dim'):
                initializer = np.zeros((self._inequality_constraints.var_dim(), ), dtype=float)
            else:
                raise RuntimeError('Unknown variable dimension. Try hinting the optimizer '
                                   'via the initialization methods')

        return opti_math.optimize(initializer, _satisfied, _director, _stepper, config, diary)

    def _constrained_initialize(self, initializer, config, opti_math, diary):
        """

        :param initializer:
        :param diary:
        :return: initializer
        :rtype: np.ndarray
        """
        initializer = self.__equality_initialize(initializer, config, opti_math, diary)
        return self.__inequality_initialize(initializer, config, opti_math, diary)

    def _direction(self, k, x_k, d_k, opti_math, diary):
        """

        :param x_k:
        :param d_k:
        :param opti_math:
        :param diary:
        :return: modified direction, usually projected onto constraints.
        :rtype: np.array
        """
        # Project d_k sequentially to the innermost active inequality constraint, then all equality constraints
        for c in [self._inequality_constraints, self._bound_constraints, self._working_equality_constraints]:
            d_k = c.direction(x_k, d_k, opti_math, diary)
        return d_k

    def _steplength(self, k, x_k, d_k, max_steplength, opti_math, diary):
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
        delta = max_steplength
        for c in [self._inequality_constraints, self._bound_constraints, self._working_equality_constraints]:
            delta = c.steplength(k, x_k, d_k, delta, opti_math, diary)
        return delta


class IntegerConstraintsMixIn(object):
    """
    """

    def __init__(self, proto_problem):
        self._integer_vars = []
        self._binary_vars = []

        for msg in proto_problem.integer_constraints:
            if np.ceil(msg.lower) > np.floor(msg.upper):
                raise ValueError('Infeasible bound for integer variable index {}: [{}, {}]'.format(
                    msg.var_index, msg.lower, msg.upper))
            self._integer_vars.append(base.BoundedVariable(
                index=msg.var_index, lower_bound=msg.lower, upper_bound=msg.upper))
        for msg in proto_problem.binary_constraints:
            self._binary_vars.append(msg)
        self._integer_var_indices = []
        self._integer_var_lowers = []
        self._integer_var_uppers = []

    """
    Called by the MIP Solver
    """

    def _setup(self, config, diary):
        """
        Actually materialize the integerity constraints.
        Not doing this in the constructor since it needs some config and diary.
        """
        if not set(self._binary_vars).isdisjoint(v.index for v in self._integer_vars):
            raise ValueError('Easy your life by disentangling binary variables from integer variables')

        filtered_vars = []
        for var_index in set([v.index for v in self._integer_vars]):
            reps = [v for v in self._integer_vars if v.index == var_index]
            lower_bound = np.max([v.lower_bound for v in reps])
            upper_bound = np.min([v.upper_bound for v in reps])
            if np.ceil(lower_bound) > np.floor(upper_bound):
                raise ValueError('Infeasible bound for integer variable index {}: [{}, {}]'.format(
                    var_index, lower_bound, upper_bound))
            if len(reps) > 1:
                items = dict()
                items['msg_integer_var_{}'.format(var_index)] = 'Reset variable {} bounds to be [{}, {}]'.format(
                    var_index, lower_bound, upper_bound)
                diary.set_items(**items)
            filtered_vars.append(base.BoundedVariable(var_index, lower_bound, upper_bound))
        self._integer_vars = filtered_vars
        self._integer_var_indices = [v.index for v in filtered_vars]
        self._integer_var_lowers = [v.lower_bound for v in filtered_vars]
        self._integer_var_uppers = [v.upper_bound for v in filtered_vars]
        self._binary_vars = sorted(set(self._binary_vars))

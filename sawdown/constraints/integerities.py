import numpy as np

from sawdown.constraints import base, inequalities, equalities


class IntegerityConstraintsBase(object):

    def range_constraints(self, var_dim=-1):
        raise NotImplementedError()

    def split(self, var_idx, bound, leq=True):
        raise NotImplementedError()

    def filter_fixed_values(self):
        raise NotImplementedError()


class EmptyIntegerityConstraints(base.EmptyConstraints, IntegerityConstraintsBase):
    def clone(self):
        return EmptyIntegerityConstraints()

    def filter_fixed_values(self):
        return base.EmptyConstraints()


class IntegerityConstraints(base.ConstraintsBase, IntegerityConstraintsBase):
    def __init__(self, variables, total_dim=-1):
        if any(np.ceil(v.lower_bound) > np.floor(v.upper_bound) for v in variables):
            raise ValueError('Invalid lower bound and upper bound for integer variables')
        if total_dim >= 0 and any(v.index >= total_dim for v in variables):
            raise ValueError('total_dim has to be at least greater than the max. variable index')

        filtered_vars = []
        for var_index in set([v.index for v in variables]):
            reps = [v for v in variables if v.index == var_index]
            lower_bound = np.max([v.lower_bound for v in reps])
            upper_bound = np.min([v.upper_bound for v in reps])
            if np.ceil(lower_bound) > np.floor(upper_bound):
                raise ValueError('Infeasible bound for variable index {}: [{}, {}]'.format(
                    var_index, lower_bound, upper_bound))
            if len(reps) > 1:
                # TODO: properly write into diaries.
                print('Reset variable {} bounds to be [{}, {}]'.format(var_index, lower_bound, upper_bound))
            filtered_vars.append(base.BoundedVariable(var_index, lower_bound, upper_bound))
        self._variables = filtered_vars

        self._total_dim = total_dim
        self._opti_math = None
        self._indices = [v.index for v in self._variables]
        self._lower_bounds = np.asarray([v.lower_bound for v in self._variables], dtype=float)
        self._upper_bounds = np.asarray([v.upper_bound for v in self._variables], dtype=float)

    def var_dim(self):
        return self._total_dim

    def indices(self):
        return self._indices[:]

    def clone(self):
        c = IntegerityConstraints([v.clone() for v in self._variables], self._total_dim)
        c.setup(None, self._opti_math)
        return c

    def setup(self, objective, opti_math, **kwargs):
        self._opti_math = opti_math

    def merge(self, other):
        assert isinstance(other, IntegerityConstraints)
        if self._total_dim != other._total_dim and self._total_dim != -1 and other._total_dim != -1:
            raise ValueError('Cannot merge 2 constraints of different variable dimensions')
        dim = self._total_dim if self._total_dim != -1 else other._total_dim
        return IntegerityConstraints([v.clone() for v in self._variables + other._variables], dim)

    def satisfied(self, x):
        values = x[self._indices]
        return np.all(np.logical_and(self._opti_math.equals(values, np.round(values)),
                                     self._opti_math.in_bounds(values, self._lower_bounds, self._upper_bounds)))

    def range_constraints(self, var_dim=-1):
        """
        Convert the integer constraints into a bunch of range constraints, which is used in the initial relaxation.
        :param var_dim:
        :return:
        """
        if not any(map(lambda _v: np.isfinite(_v.lower_bound) or np.isfinite(_v.upper_bound), self._variables)):
            return base.EmptyConstraints()

        var_dim = var_dim if var_dim > 0 else self.var_dim()
        if var_dim < 0:
            raise ValueError('Unknown variable dimension')
        constraint_a = []
        constraint_b = []
        for v in self._variables:
            if np.isfinite(v.lower_bound):
                a = np.zeros((1, var_dim), dtype=float)
                a[0, v.index] = 1.
                constraint_a.append(a)
                constraint_b.append(-v.lower_bound)
            if np.isfinite(v.upper_bound):
                a = np.zeros((1, var_dim), dtype=float)
                a[0, v.index] = -1.
                constraint_a.append(a)
                constraint_b.append(v.upper_bound)

        c = inequalities.LinearInequalityConstraints(np.vstack(constraint_a),
                                                     np.asarray(constraint_b, dtype=float))
        c.setup(None, self._opti_math)
        return c

    def split(self, var_idx, bound, leq=True):
        """
        Return the set of inequality (in form of IntegerityConstraints) and fixed-value constraints
        occurred when splitting the given variable index.
        """
        if not np.isfinite(bound):
            raise ValueError('Food and water for you.')

        variable = next((v for v in self._variables if v.index == var_idx), None)
        if variable is None:
            raise ValueError('variable #{} is not an integer one'.format(var_idx))
        # integer_constraints = IntegerityConstraints(
        #     [v.clone() for v in self._variables if v.index != var_idx],
        #     total_dim=self._total_dim)
        integer_constraints = EmptyIntegerityConstraints()

        fixed_value = base.EmptyConstraints()
        if leq:
            # x <= bound
            if self._opti_math.equals(bound, variable.lower_bound):
                fixed_value = equalities.FixedValueConstraints(
                    [base.FixedVariable(variable.index, bound)], self._total_dim)
            else:
                assert self._opti_math.lt(variable.lower_bound, bound), 'This is bad.'
                new_constraints = IntegerityConstraints(
                    [base.BoundedVariable(variable.index, variable.lower_bound, bound)], self._total_dim)
                integer_constraints = integer_constraints.merge(new_constraints)
        else:
            # x >= bound
            if self._opti_math.equals(bound, variable.upper_bound):
                fixed_value = equalities.FixedValueConstraints(
                    [base.FixedVariable(variable.index, bound)], self._total_dim)
            else:
                assert self._opti_math.lt(bound, variable.upper_bound), 'This is bad, too.'
                new_constraints = IntegerityConstraints(
                    [base.BoundedVariable(variable.index, bound, variable.upper_bound)], self._total_dim)
                integer_constraints = integer_constraints.merge(new_constraints)
        return integer_constraints, fixed_value

    def filter_fixed_values(self):
        """
        Returns a tuple of filtered integer constraints and fixed-value constraints.
        Fixed-value constraints happens when branching multiple times on the same variable index.

        Should be called after each call to `.merge()`,
        and need to `.setup(..., opti_math, ...)` before calling this function.

        :rtype: (IntegerityConstraints, equalities.FixedValueConstraints)
        """
        fixed_vars = []
        integer_vars = []
        for var_index in set(self._indices):
            reps = [v for v in self._variables if v.index == var_index]
            lower_bound = np.ceil(np.max([v.lower_bound for v in reps]))
            upper_bound = np.floor(np.min([v.upper_bound for v in reps]))
            assert lower_bound <= upper_bound
            if self._opti_math.equals(lower_bound, upper_bound):
                fixed_vars.append(base.FixedVariable(var_index, lower_bound))
            else:
                integer_vars.append(base.BoundedVariable(var_index, lower_bound, upper_bound))
        integerity_constraints = (EmptyIntegerityConstraints() if len(integer_vars) == 0 else
                                  IntegerityConstraints(integer_vars, self._total_dim))
        fixed_value_constraints = (base.EmptyConstraints() if len(fixed_vars) == 0 else
                                   equalities.FixedValueConstraints(fixed_vars, self._total_dim))
        return integerity_constraints, fixed_value_constraints

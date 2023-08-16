import numpy as np

from sawdown import constraints
from sawdown.proto import sawdown_pb2, serializer
from sawdown.optimizers import base, branch_and_bound


class MipOptimizer(base.OptimizerBase, constraints.IntegerConstraintsMixIn, branch_and_bound.BranchAndBounder):
    """
    Solving MIPs via a Multi-process Branch-and-bound.
    """

    def __init__(self, proto_problem):
        base.OptimizerBase.__init__(self, proto_problem)
        constraints.IntegerConstraintsMixIn.__init__(self, proto_problem)
        branch_and_bound.BranchAndBounder.__init__(self, proto_problem)
        self._var_dim = -1

    def _initial_problems(self, diary):
        if self._objective is None:
            raise ValueError('Missing objective')
        if len(self._stoppers) == 0:
            raise ValueError('Missing stoppers: Machine Power is priceless, do not waste.')

        # Run setup() just to check for validity.
        constraints.IntegerConstraintsMixIn._setup(self, self._config, diary)
        if len(self._integer_vars) == 0 and len(self._binary_vars) == 0:
            raise ValueError('Specify some integer/binary constraints, or use FirstOrderOptimizer instead')

        initial_problem = sawdown_pb2.IntegerSubproblem()

        # merge the integer and binary constraints into bound constraints.
        existing_bounded_vars = set(v.index for v in self._proto_problem.bound_constraints)
        for integer_var in self._integer_vars:
            if integer_var.index in existing_bounded_vars:
                raise ValueError('Integer variable #{} has duplicated bound constraints'.format(integer_var.index))
            existing_bounded_vars.add(integer_var.index)
            initial_problem.bound_constraints.append(sawdown_pb2.BoundConstraint(
                var_index=integer_var.index, lower=integer_var.lower_bound, upper=integer_var.upper_bound))

        binary_var_lower, binary_var_upper = self._config.binary_bounds()
        for binary_var_index in self._binary_vars:
            if binary_var_index in existing_bounded_vars:
                raise ValueError('Binary variable #{} has duplicated bound constraints'.format(binary_var_index))
            existing_bounded_vars.add(binary_var_index)
            initial_problem.bound_constraints.append(sawdown_pb2.BoundConstraint(
                var_index=binary_var_index, lower=binary_var_lower, upper=binary_var_upper))

        initializer = None
        for init in self._initializers:
            initializer = init.initialize(initializer)

        if initializer is not None:
            initial_problem.initializer = serializer.encode_ndarray(initializer)
        initial_problem.diary_id = diary.new_sub_id()

        return [initial_problem]

    def _accept(self, solution, diary):
        if len(self._integer_vars) > 0:
            value = solution.x[self._integer_var_indices]
            assert self._opti_math.in_bounds(value, self._integer_var_lowers, self._integer_var_uppers)
            if not self._opti_math.equals(value, np.round(value)):
                return False

        if len(self._binary_vars) > 0:
            bounds = self._config.binary_bounds()
            value = solution.x[self._binary_vars]
            if not(self._opti_math.equals(value, bounds[0])) and not(self._opti_math.equals(value, bounds[1])):
                return False
        return True

    def _branch(self, sub_problem, sub_solution, diary):
        # TODO: make picking split_idx customizable.
        x = sub_solution.x
        assert x.ndim == 1
        integer_residuals = np.square(x[self._integer_var_indices] - np.floor(x[self._integer_var_indices]) - 0.5)
        binary_bounds = self._config.binary_bounds()
        binary_mid = 0.5 * (binary_bounds[1] - binary_bounds[0])
        binary_residuals = np.square(x[self._binary_vars] - binary_bounds[0] - binary_mid)
        split_idx = np.argmin(np.concatenate((integer_residuals, binary_residuals)))

        binary_var = split_idx >= integer_residuals.size
        if binary_var:
            split_idx = self._binary_vars[split_idx - integer_residuals.size]
        else:
            split_idx = self._integer_var_indices[split_idx]

        assert split_idx in set(v.var_index for v in sub_problem.bound_constraints)
        assert split_idx not in set(v.var_index for v in sub_problem.fixed_value_constraints)
        template_problem = sawdown_pb2.IntegerSubproblem()
        # copy the existing bound constraints, except split_idx
        for var in sub_problem.bound_constraints:
            if var.var_index != split_idx:
                copied = sawdown_pb2.BoundConstraint()
                copied.CopyFrom(var)
                template_problem.bound_constraints.append(copied)
        # copy the existing fixed value constraints
        for var in sub_problem.fixed_value_constraints:
            copied = sawdown_pb2.FixedValueConstraint()
            copied.CopyFrom(var)
            template_problem.fixed_value_constraints.append(copied)

        sub_problems = []
        if binary_var:
            for bound in binary_bounds:
                new_problem = sawdown_pb2.IntegerSubproblem()
                new_problem.CopyFrom(template_problem)
                new_problem.fixed_value_constraints.append(
                    sawdown_pb2.FixedValueConstraint(var_index=split_idx, value=bound))

                initializer = x.copy()
                initializer[split_idx] = bound
                new_problem.initializer.CopyFrom(serializer.encode_ndarray(initializer))
                new_problem.diary_id = diary.new_sub_id()
                sub_problems.append(new_problem)
        else:
            integer_var = next(v for v in self._integer_vars if v.index == split_idx)

            for d, bound in enumerate((np.floor(x[split_idx]), np.floor(x[split_idx]) + 1)):
                new_problem = sawdown_pb2.IntegerSubproblem()
                new_problem.CopyFrom(template_problem)
                if d == 0:
                    # x <= bound
                    if self._opti_math.equals(integer_var.lower_bound, bound):
                        new_problem.fixed_value_constraints.append(
                            sawdown_pb2.FixedValueConstraint(var_index=split_idx, value=bound))
                    else:
                        assert self._opti_math.lt(integer_var.lower_bound, bound)
                        new_problem.bound_constraints.append(sawdown_pb2.BoundConstraint(
                            var_index=split_idx, lower=integer_var.lower_bound, upper=bound))
                else:
                    # x >= bound
                    if self._opti_math.equals(integer_var.upper_bound, bound):
                        new_problem.fixed_value_constraints.append(
                            sawdown_pb2.FixedValueConstraint(var_index=split_idx, value=bound))
                    else:
                        assert self._opti_math.lt(bound, integer_var.upper_bound)
                        new_problem.bound_constraints.append(sawdown_pb2.BoundConstraint(
                            var_index=split_idx, lower=bound, upper=integer_var.upper_bound))

                initializer = x.copy()
                initializer[split_idx] = bound
                new_problem.initializer.CopyFrom(serializer.encode_ndarray(initializer))
                new_problem.diary_id = diary.new_sub_id()
                sub_problems.append(new_problem)

        return sub_problems

import numpy as np
from sawdown import meta_optimizers, errors, opti_math, diaries, objectives, constraints, initializers, directions, \
    steplengths, stoppers, first_orders


class MipProblemStatement(object):
    def __init__(self, integers, fixed_values, initializer=None, sub_diary=None):
        self.integer_constraints = integers
        self.fixed_value_constraints = fixed_values
        self.initializer = initializer
        self.sub_diary = sub_diary


class MipOptimizer(meta_optimizers.BranchAndBounder,
                   objectives.ObjectiveMixIn,
                   opti_math.OptiMathMixIn,
                   constraints.ConstraintsMixIn, constraints.IntegerConstraintsMixIn,
                   initializers.InitializerMixIn, directions.DirectionMixIn,
                   steplengths.SteplengthMixIn, stoppers.StopperMixIn):
    """
    Solving MIPs via a Multi-process Branch-and-bound.
    """

    def __init__(self):
        meta_optimizers.BranchAndBounder.__init__(self)
        objectives.ObjectiveMixIn.__init__(self)
        opti_math.OptiMathMixIn.__init__(self)
        constraints.ConstraintsMixIn.__init__(self)
        constraints.IntegerConstraintsMixIn.__init__(self)
        initializers.InitializerMixIn.__init__(self)
        directions.DirectionMixIn.__init__(self)
        steplengths.SteplengthMixIn.__init__(self)
        stoppers.StopperMixIn.__init__(self)
        self._var_dim = -1

    def _initial_problems(self, diary):
        if self._objective is None:
            raise ValueError('Missing objective')
        if len(self._stoppers) == 0:
            raise ValueError('Missing stoppers: Machine Power is priceless, do not waste.')
        if self._integer_constraints.is_empty():
            raise ValueError('Specify some integer constraints, or use FirstOrderOptimizer instead')
        constraints.ConstraintsMixIn._setup(self, self._objective, self._opti_math)
        constraints.IntegerConstraintsMixIn._setup(self, self._objective, self._opti_math)
        initializers.InitializerMixIn._setup(self, self._objective, self._opti_math)
        directions.DirectionMixIn._setup(self, self._objective, self._opti_math)
        steplengths.SteplengthMixIn._setup(self, self._objective, self._opti_math)
        stoppers.StopperMixIn._setup(self, self._objective, self._opti_math)
        diaries.AsyncDiaryMixIn._setup(self, self._objective, self._opti_math)

        initializer = initializers.InitializerMixIn._initialize(self)

        # Kinda trick
        original_inequalities = self._inequality_constraints.clone()
        try:
            if initializer is not None:
                self._var_dim = initializer.size
            else:
                relaxed_constraints = (self._equality_constraints, self._fixed_value_constraints,
                                       self._inequality_constraints)
                self._var_dim = next((c.var_dim() for c in relaxed_constraints if c.var_dim() != -1), -1)

            self._inequality_constraints = self._merge_constraints(
                self._inequality_constraints, self._integer_constraints.range_constraints(self._var_dim))

            with diary.sub() as sub_diary:
                sub_diary.set_items(msg='Starting initialization')
                initializer = constraints.ConstraintsMixIn._initialize(self, initializer, sub_diary)
        finally:
            self._inequality_constraints = original_inequalities

        if initializer is None:
            raise errors.InitializationError('Failed to initialize. Missing initializer, maybe.')

        self._objective.check_dimensions(initializer.size)

        relaxed_problem = MipProblemStatement(constraints.EmptyConstraints(), constraints.EmptyConstraints(),
                                              initializer, diary.sub(with_queues=False))
        return [relaxed_problem]

    def _accept(self, solution, diary):
        return self._integer_constraints.satisfied(solution.x)

    def _branch(self, problem, solution, diary):
        # TODO: make picking split_idx customizable.
        integer_vars = sorted(set(self._integer_constraints.indices()))

        x = solution.x
        split_idx = np.argmin(np.square(x[integer_vars] - np.floor(x[integer_vars]) - 0.5))
        split_idx = integer_vars[split_idx]
        diary.set_items(split_idx=split_idx)
        sub_problems = []
        for d in (0, 1):
            initializer = x.copy()
            initializer[split_idx] = np.floor(initializer[split_idx]) + d
            bound = initializer[split_idx]
            (integer_constraints, fixed_value_constraints) = self._integer_constraints.split(split_idx, bound, d == 0)

            sub_problem = MipProblemStatement(
                self._merge_constraints(problem.integer_constraints, integer_constraints),
                self._merge_constraints(problem.fixed_value_constraints, fixed_value_constraints),
                initializer, diary.sub(with_queues=False))
            sub_problems.append(sub_problem)
        return sub_problems

    def _sub_optimizer(self, problem, diary):
        integer_constraints = self._merge_constraints(self._integer_constraints, problem.integer_constraints)
        integer_constraints.setup(None, opti_math=self._opti_math)
        integer_constraints, fixed_value_constraints = integer_constraints.filter_fixed_values()
        integer_range_constraints = (None if integer_constraints.is_empty() else
                                     integer_constraints.range_constraints(problem.initializer.size))
        fixed_value_constraints = self._merge_constraints(problem.fixed_value_constraints, fixed_value_constraints)

        # Clone the constraints of self, then add the constraints in problem.
        optimizer = first_orders.FirstOrderOptimizer().set_objective(self._objective.clone()) \
            .fixed_initializer(problem.initializer) \
            .append_constraints(self._inequality_constraints,
                                self._equality_constraints, self._fixed_value_constraints) \
            .append_constraints(integer_range_constraints, None, fixed_value_constraints) \
            .direction_calculator_from(self) \
            .steplength_calculators_from(self) \
            .stoppers_from(self) \
            .config(epsilon=self._opti_math.epsilon, initialization_max_iters=self._opti_math.initialization_max_iters,
                    initialization_decay_steps=self._opti_math.initialization_decay_steps) \
            .log(problem.sub_diary)
        return optimizer

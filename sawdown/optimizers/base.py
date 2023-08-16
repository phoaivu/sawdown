import pickle

from sawdown import objectives, initializers, constraints, directions, steplengths, stoppers, config, opti_math
from sawdown.proto import sawdown_pb2, serializer


class OptimizerBase(constraints.ConstraintsMixIn):
    """
    The base of all optimizers. Containing:
    - Objective function
    - Initializer
    - Continuous Constraints: equalities, inequalities, fixed-values, bound
    - Direction calculator
    - Steplength calculators
    - Stoppers
    - Configuration
    - Relaxed-precision math ops.

    This is constructed via a protobuf definition.
    Diaries are constructed separately depending on whether the optimizer is multi- or single-process.
    """
    def __init__(self, proto_problem):
        constraints.ConstraintsMixIn.__init__(self, proto_problem)
        self._objective = None
        self._initializers = []
        self._direction_calculator = None
        self._steplength_calculators = []
        self._stoppers = []
        self._config = config.Config()
        self._opti_math = opti_math.OptiMath()

        field_name = proto_problem.WhichOneof('objective')
        if field_name == 'python_func_objective':
            self._objective = objectives.FirstOrderObjective(
                objective_func=pickle.loads(proto_problem.python_func_objective.objective),
                deriv_func=pickle.loads(proto_problem.python_func_objective.gradient))
        elif field_name == 'instance_objective':
            self._objective = serializer.decode_method(proto_problem.instance_objective)
        else:
            raise ValueError('Undefined objective')

        for proto_initializer in proto_problem.initializers:
            field_name = proto_initializer.WhichOneof('initializer')
            if field_name == 'fixed_initializer':
                self._initializers.append(initializers.FixedInitializer(
                    serializer.decode_ndarray(proto_initializer.fixed_initializer)))
            elif field_name == 'uniform_initializer':
                self._initializers.append(initializers.UniformInitializer(
                    var_dim=proto_initializer.uniform_initializer.var_dim,
                    low=proto_initializer.uniform_initializer.low,
                    high=proto_initializer.uniform_initializer.high))
            elif field_name is not None:
                raise ValueError('Unsupported initializer: {}'.format(field_name))

        field_name = proto_problem.WhichOneof('direction_calculator')
        if field_name == 'steepest_descent':
            self._direction_calculator = directions.SteepestDecent()
        elif field_name == 'conjugate_gradient':
            self._direction_calculator = directions.ConjugateGradient(beta=proto_problem.conjugate_gradient.beta)
        elif field_name == 'adam':
            self._direction_calculator = directions.Adam(alpha=proto_problem.adam.alpha,
                                                         beta=proto_problem.adam.beta)
        elif field_name is not None:
            raise ValueError('Unsupported direction calculator: {}'.format(field_name))
        if self._direction_calculator is None:
            raise ValueError('Missing direction calculator')

        for proto_steplength in proto_problem.steplength_calculators:
            field_name = proto_steplength.WhichOneof('steplength_calculator')
            if field_name == 'quadratic_interpolation':
                self._steplength_calculators.append(steplengths.QuadraticInterpolationSteplength())
            elif field_name == 'decayed_steplength':
                self._steplength_calculators.append(steplengths.DecaySteplength(
                    decay_steps=proto_steplength.decayed_steplength.decay_steps))
            elif field_name == 'circle_detection':
                self._steplength_calculators.append(steplengths.CircleDetectionSteplength(
                    circle_length=proto_steplength.circle_detection.circle_length))
            elif field_name is not None:
                raise ValueError('Unsupported steplength calculator: {}'.format(field_name))

        for proto_stopper in proto_problem.stoppers:
            field_name = proto_stopper.WhichOneof('stopper')
            if field_name == 'max_iters_stopper':
                self._stoppers.append(stoppers.MaxIterationsStopper(
                    max_iters=proto_stopper.max_iters_stopper.max_iters))
            elif field_name == 'infinitesimal_step_stopper':
                self._stoppers.append(stoppers.InfinitesimalStepStopper())
            elif field_name is not None:
                raise ValueError('Unsupported stopper: {}'.format(field_name))
        if len(self._stoppers) == 0:
            raise ValueError('Need at least a stopper')

        if proto_problem.HasField('config'):
            proto_config = proto_problem.config
            if proto_config.HasField('epsilon'):
                self._opti_math.epsilon = proto_config.epsilon
            if proto_config.HasField('binary_mapping_mode'):
                if proto_config.binary_mapping_mode == sawdown_pb2.BINARY_MAPPING_ZERO_ONE:
                    self._config.binary_mapping_mode = config.BinaryMappingMode.ZERO_ONE
                elif proto_config.binary_mapping_mode == sawdown_pb2.BINARY_MAPPING_ONES:
                    self._config.binary_mapping_mode = config.BinaryMappingMode.ONES
                elif proto_config.binary_mapping_mode != sawdown_pb2.BINARY_MAPPING_UNSPECIFIED:
                    raise ValueError('Unsupported binary mapping mode: {}'.format(
                        repr(proto_config.binary_mapping_mode)))
            for field_name in ['initialization_max_iters', 'initialization_decay_steps', 'parallelization']:
                if proto_config.HasField(field_name):
                    setattr(self._config, field_name, getattr(proto_config, field_name))
        self._opti_math.check_precision()

        # Additional checks, for ill-informed users.
        if (len(set(v.var_index for v in proto_problem.fixed_value_constraints))
                < len(proto_problem.fixed_value_constraints)):
            raise ValueError('Duplicated entries in fixed value constraints')

        for i, var in enumerate(proto_problem.bound_constraints):
            if len([v for v in proto_problem.fixed_value_constraints if v.var_index == var.var_index]) > 0:
                raise ValueError('Variable #{} has both fixed value and bound constraints'.format(var.var_index))
            if self._opti_math.equals(var.lower, var.upper):
                raise ValueError('Variable #{} has singleton bounds. Use fixed-value constraint instead'.format(
                    var.var_index))
            if len([v for v in proto_problem.bound_constraints[i+1:] if v.var_index == var.var_index]) > 0:
                raise ValueError('Variable #{} has duplicated bound constraints'.format(var.var_index))

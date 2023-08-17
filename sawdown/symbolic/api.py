import os.path
import pickle

import numpy as np

from sawdown import objectives, optimizers
from sawdown.proto import sawdown_pb2, serializer


class Declaration(object):
    def __init__(self):
        self._problem = sawdown_pb2.Problem()

    def objective_functors(self, objective, grad):
        self._problem.python_func_objective.objective = serializer.encode_functor(objective)
        self._problem.python_func_objective.gradient = serializer.encode_functor(grad)
        return self

    def objective_instance(self, objective_class, *args):
        if not issubclass(objective_class, objectives.ObjectiveBase):
            raise ValueError('Need to be a subclass of ObjectiveBase')
        self._problem.instance_objective.CopyFrom(serializer.encode_method(objective_class, *args))
        return self

    def fixed_initializer(self, initializer):
        self._problem.initializers.append(sawdown_pb2.Initializer(
            fixed_initializer=serializer.encode_ndarray(initializer)))
        return self

    def constant_initializer(self, var_dim=1, constant=0.):
        """
        Initialize with a vector of dimension `var_dim` containing all `constant`. Both arguments are scalars.
        """
        return self.fixed_initializer(np.ones((var_dim, ), dtype=float) * constant)

    def zeros_initializer(self, var_dim=1):
        return self.fixed_initializer(np.zeros((var_dim, ), dtype=float))

    def uniform_initializer(self, var_dim=1, low=0., high=1.):
        self._problem.initializers.append(sawdown_pb2.Initializer(
            uniform_initializer=sawdown_pb2.UniformInitializer(var_dim=var_dim, low=low, high=high)))
        return self

    # Constraints

    def linear_inequality_constraints(self, a, b):
        """
        ax + b >= 0.
        """
        self._problem.linear_inequalities.append(sawdown_pb2.LinearConstraint(
            a=serializer.encode_ndarray(a), b=serializer.encode_ndarray(b)))
        return self

    def linear_equality_constraints(self, a, b):
        """
        ax + b = 0
        """
        self._problem.linear_equalities.append(sawdown_pb2.LinearConstraint(
            a=serializer.encode_ndarray(a), b=serializer.encode_ndarray(b)))
        return self

    def fixed_value_constraint(self, var_index=0, value=0.):
        """
        The given variable take pre-described value.
        End-users normally do not use this (because you don't need to optimize a fixed variable!),
        it is used mainly to speed-up binary and integer programs.
        """
        self._problem.fixed_value_constraints.append(sawdown_pb2.FixedValueConstraint(var_index=var_index, value=value))
        return self

    def bound_constraint(self, var_index=0, lower=-np.inf, upper=np.inf):
        return self.bound_constraints((var_index,), (lower, ), (upper, ))

    def bound_constraints(self, var_indices=(), lower_bounds=(), upper_bounds=()):
        lower_bounds = tuple(lower_bounds)
        upper_bounds = tuple(upper_bounds)
        if len(set(var_indices)) < len(var_indices):
            raise ValueError('Duplicated variable indices')
        if len(lower_bounds) == 1 and len(var_indices) > 1:
            lower_bounds = [lower_bounds[0] for _ in var_indices]
        if len(upper_bounds) == 1 and len(var_indices) > 1:
            upper_bounds = [upper_bounds[0] for _ in var_indices]
        for index, lower, upper in zip(var_indices, lower_bounds, upper_bounds):
            self._problem.bound_constraints.append(
                sawdown_pb2.BoundConstraint(var_index=index, lower=lower, upper=upper))
        return self

    def steepest_descent(self):
        self._problem.steepest_descent.CopyFrom(sawdown_pb2.SteepestDescent())
        return self

    def conjugate_gradient(self, beta=0.9):
        self._problem.conjugate_gradient.CopyFrom(sawdown_pb2.ConjugateGradient(beta=beta))
        return self

    def adam(self, alpha=0.9, beta=0.99):
        self._problem.adam.CopyFrom(sawdown_pb2.Adam(alpha=alpha, beta=beta))
        return self

    # Steplength calculators

    def decayed_steplength(self, decay_steps=100):
        self._problem.steplength_calculators.append(
            sawdown_pb2.Steplength(decayed_steplength=sawdown_pb2.DecayedSteplength(decay_steps=decay_steps)))
        return self

    def quadratic_interpolation_steplength(self):
        self._problem.steplength_calculators.append(
            sawdown_pb2.Steplength(quadratic_interpolation=sawdown_pb2.QuadraticInterpolation()))
        return self

    def circle_detection_steplength(self, circle_length=2, decay_rate=0.5):
        """
        A form of conditional decayed steplength. The steplength is decayed by a factor of `decay_rate`
        everytime the optimizer goes in circles of length `circle_length`.
        """
        self._problem.steplength_calculators.append(sawdown_pb2.Steplength(
            circle_detection=sawdown_pb2.CircleDetection(circle_length=circle_length, decay_rate=decay_rate)))
        return self

    def stop_after(self, max_iters=1000):
        self._problem.stoppers.append(
            sawdown_pb2.Stopper(max_iters_stopper=sawdown_pb2.MaxIterationsStopper(max_iters=max_iters)))
        return self

    def stop_small_steps(self):
        self._problem.stoppers.append(
            sawdown_pb2.Stopper(infinitesimal_step_stopper=sawdown_pb2.InfinitesimalStepStopper()))
        return self

    def config(self, **kwargs):
        for name, val in kwargs.items():
            try:
                setattr(self._problem.config, name, val)
            except AttributeError:
                raise ValueError('Configuration name not found: {}'.format(name))
            except TypeError:
                raise ValueError('Invalid type for {}: expected {}, got {}'.format(
                    name, type(getattr(self._problem.config, name)), type(val)))
        return self

    def config_initialization(self, max_iters=1000, decay_steps=100):
        return self.config(initialization_max_iters=max_iters, initialization_decay_steps=decay_steps)

    def epsilon(self, epsilon=1e-28):
        return self.config(epsilon=epsilon)

    # Diary config
    def diary(self):
        """
        Returned Solution will have in-memory iteration data.
        """
        self._problem.diaries.append(sawdown_pb2.Diary(memory_diary=sawdown_pb2.MemoryDiary()))
        return self

    def stream_diary(self, stream='stdout'):
        """
        Returned Solution doesn't hold iteration data.
        """
        if stream not in {'stdout', 'stderr'}:
            raise ValueError('stream has to be `stdout` or `stderr`')
        self._problem.diaries.append(sawdown_pb2.Diary(stream_diary=sawdown_pb2.StreamDiary(stream=stream)))
        return self

    def file_diary(self, path='.', job_name='optimization'):
        """
        Returned Solution contains a reader configuration.
        """
        self._problem.diaries.append(sawdown_pb2.Diary(file_diary=sawdown_pb2.FileDiary(
            path=os.path.abspath(path), job_name=job_name)))
        return self

    def optimize(self):
        raise NotImplementedError()


class FirstOrderOptimizer(Declaration):

    def optimize(self):
        return optimizers.FirstOrderOptimizer(self._problem).optimize()


class MipOptimizer(Declaration):

    def integer_constraint(self, var_index=0, lower_bound=-np.inf, upper_bound=np.inf):
        return self.integer_constraints((var_index,), (lower_bound,), (upper_bound,))

    def integer_constraints(self, var_indices=(), lower_bound=(), upper_bound=()):
        lower_bound = tuple(lower_bound)
        upper_bound = tuple(upper_bound)
        if len(set(var_indices)) < len(var_indices):
            raise ValueError('Duplicated variable indices')
        if len(lower_bound) == 1 and len(var_indices) > 1:
            lower_bound = [lower_bound[0] for _ in var_indices]
        if len(upper_bound) == 1 and len(var_indices) > 1:
            upper_bound = [upper_bound[0] for _ in var_indices]
        for index, lower, upper in zip(var_indices, lower_bound, upper_bound):
            self._problem.integer_constraints.append(
                sawdown_pb2.BoundConstraint(var_index=index, lower=lower, upper=upper))
        return self

    def binary_constraint(self, var_index=0):
        return self.binary_constraints((var_index, ))

    def binary_constraints(self, var_indices=()):
        for i in sorted(set(var_indices)):
            self._problem.binary_constraints.append(i)
        return self

    def binary_mapping(self, mode='zero-ones'):
        """
        This effects how objective function is written.
        """
        accepted_modes = ['zero-ones', 'ones']
        if mode not in accepted_modes:
            raise ValueError('Accepted modes are: {}'.format(accepted_modes))
        if mode == accepted_modes[0]:
            return self.config(binary_mapping_mode=sawdown_pb2.BINARY_MAPPING_ZERO_ONE)
        return self.config(binary_mapping_mode=sawdown_pb2.BINARY_MAPPING_ONES)

    def parallelize(self, n_process=os.cpu_count()):
        if n_process < 0:
            raise ValueError('Cannot solve your problem with negative worker')
        return self.config(parallelization=n_process)

    def optimize(self):
        return optimizers.MipOptimizer(self._problem).optimize()

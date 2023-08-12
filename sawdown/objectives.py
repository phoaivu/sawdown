import numpy as np


class ObjectiveMixIn(object):
    def __init__(self):
        self._objective = None

    def set_objective(self, objective):
        """
        Replace the objective by the given one.
        :type objective: ObjectiveBase
        :return: self
        """
        self._objective = objective
        return self

    def objective(self, objective_func, deriv_func):
        self._objective = FirstOrderObjective(objective_func=objective_func, deriv_func=deriv_func)
        return self


class ObjectiveBase(object):

    def clone(self):
        return self.__class__()

    def _objective(self, x):
        raise NotImplementedError()

    def _gradient(self, x):
        raise NotImplementedError()

    def objective(self, x):
        return self._objective(x[:, None] if x.ndim == 1 else x).squeeze()

    def deriv_variables(self, x):
        if x.ndim == 1:
            return self._gradient(x[:, None]).squeeze(axis=1)
        return self._gradient(x)

    def check_dimensions(self, var_dim):
        # checking signature and dimensions of the objective function and derivative computation.
        if self.objective(np.zeros((var_dim,), dtype=float)).size != 1:
            raise ValueError('Objective function must take a {}x1 matrix and return a 1x1 matrix'.format(var_dim))
        if self.objective(np.zeros((var_dim, 5), dtype=float)).shape != (5,):
            raise ValueError('Objective function takes a {}xK matrix and returns a 1xK matrix'.format(var_dim))

        if self.deriv_variables(np.zeros((var_dim,), dtype=float)).shape != (var_dim,):
            raise ValueError('Derivative w.r.t. variables takes a {}x1 matrix and return a {}x1 matrix'.format(
                var_dim, var_dim))
        if self.deriv_variables(np.zeros((var_dim, var_dim + 1), dtype=float)).shape != (var_dim, var_dim + 1):
            raise ValueError('Derivative w.r.t. variables takes a {}xK matrix and return a {}xK matrix'.format(
                var_dim, var_dim))


class FirstOrderObjective(ObjectiveBase):
    def __init__(self, objective_func, deriv_func):
        if objective_func is None or deriv_func is None:
            raise ValueError('Null functors')
        self._objective_func = objective_func
        self._deriv_func = deriv_func

    def clone(self):
        return FirstOrderObjective(self._objective_func, self._deriv_func)

    def _objective(self, x):
        return self._objective_func(x)

    def _gradient(self, x):
        return self._deriv_func(x)

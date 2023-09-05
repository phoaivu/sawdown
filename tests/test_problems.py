import os.path
import glob

import numpy as np
import sawdown


log_path = os.path.join(os.path.split(__file__)[0], 'logs')


def cleanup_file_diary(job_name='test'):
    if os.path.exists(os.path.join(log_path, job_name)):
        [os.remove(f) for f in glob.glob(os.path.join(log_path, job_name, '*.*'))]
        os.rmdir(os.path.join(log_path, job_name))


class KnapsackObjective(sawdown.ObjectiveBase):
    def __init__(self, values=None):
        sawdown.ObjectiveBase.__init__(self)
        self.values = values

    def _objective(self, x):
        return -np.matmul(self.values[None, :], x)

    def _gradient(self, x):
        return -np.matmul(self.values[:, None], np.ones((1, x.shape[1]), dtype=float))


class LeastSquareObjective(sawdown.ObjectiveBase):

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def _objective(self, x):
        return 0.5 * np.sum(np.square(np.matmul(self._a, x) - self._b[:, None]), axis=0)

    def _gradient(self, x):
        return np.matmul(self._a.T, np.matmul(self._a, x) - self._b[:, None])


class SimpleInequalityQuadraticObjective(sawdown.ObjectiveBase):

    def __init__(self):
        pass

    def _objective(self, x):
        # (x + 2)^2 + (y+2)^2 = x^2 + y^2 + 4(x+y)
        return np.sum(np.square(x), axis=0) + 4. * np.sum(x, axis=0)

    def _gradient(self, x):
        return 2. * x + 4.


def knapsack():
    values = np.asarray([5., 2., 3., 6., 8.], dtype=float)
    weights = np.asarray([[3., 4., 6., 2., 5.]], dtype=float)
    max_weights = np.asarray([17.], dtype=float)

    return sawdown.MipOptimizer().objective_instance(KnapsackObjective, values) \
        .linear_inequality_constraints(-weights, max_weights) \
        .binary_constraints(var_indices=list(range(5))) \
        .steepest_descent().decayed_steplength(10) \
        .stop_after(100).stop_small_steps() \
        .config(epsilon=1e-2) \
        .parallelize(4)


def knapsack_fault():
    """
    Wrongly configured knapsack problem: specify bound constraint for a binary variable
    """
    values = np.asarray([5., 2., 3., 6., 8.], dtype=float)
    weights = np.asarray([[3., 4., 6., 2., 5.]], dtype=float)
    max_weights = np.asarray([17.], dtype=float)

    return sawdown.MipOptimizer().objective_instance(KnapsackObjective, values) \
        .linear_inequality_constraints(-weights, max_weights) \
        .bound_constraint(var_index=0, lower=0, upper=1) \
        .binary_constraints(var_indices=list(range(5))) \
        .steepest_descent().decayed_steplength(10) \
        .stop_after(100).stop_small_steps() \
        .config(epsilon=1e-2) \
        .parallelize(4)


def least_square(_a, _b):
    return sawdown.FirstOrderOptimizer().objective_instance(LeastSquareObjective, _a, _b) \
        .fixed_initializer(np.zeros_like(_a[0, :], dtype=float)) \
        .steepest_descent() \
        .quadratic_interpolation_steplength() \
        .stop_after(100).stop_small_steps()


def simple_inequality_quadratic():
    constraint_a = np.asarray([[1., 0.], [0., 1.]], dtype=float)
    constraint_b = np.asarray([1., 0.], dtype=float)

    return sawdown.FirstOrderOptimizer().objective_instance(SimpleInequalityQuadraticObjective) \
        .linear_inequality_constraints(constraint_a, constraint_b) \
        .steepest_descent().quadratic_interpolation_steplength() \
        .stop_after(100).stop_small_steps()


class QueenObjective(sawdown.ObjectiveBase):

    def _objective(self, x):
        return -0.5 * np.sum(np.square(x), axis=0)

    def _gradient(self, x):
        return -x


def queens(n=4):
    # horizontal and vertical
    a = np.zeros((2 * n, n * n), dtype=float)
    b = np.ones((2 * n, ), dtype=float)
    for i in range(n):
        a[i, i * n:(i + 1) * n] = 1.
        a[n + i, np.arange(i, n * n, n)] = 1.
    # // and \\
    c = np.zeros((4 * n - 2, n * n), dtype=float)
    d = np.ones((4 * n - 2,), dtype=float)
    for i in range(n):
        c[i, np.arange(i, i*n + 1, n-1)] = 1.
        c[2 * n - 1 + i, np.arange(i, i + (n-i)*(n+1), n + 1)] = 1.
    for i in range(n-1):
        c[n + i, np.arange(n*(i+2) - 1, n * n, n - 1)] = 1.
        c[3 * n - 1 + i, np.arange(n*(i+1), n*(i+1) + (n-i-1)*(n+1), n + 1)] = 1.

    return sawdown.MipOptimizer().objective_instance(QueenObjective) \
        .fixed_initializer(np.ones(n*n)) \
        .linear_equality_constraints(a, -b) \
        .binary_constraints(tuple(range(n * n))) \
        .steepest_descent().quadratic_interpolation_steplength() \
        .stop_after(100).stop_small_steps()

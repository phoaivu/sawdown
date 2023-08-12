import numpy as np
import sawdown


class KnapsackObjective(sawdown.ObjectiveBase):
    def __init__(self, values=None):
        sawdown.ObjectiveBase.__init__(self)
        self.values = values

    def clone(self):
        return KnapsackObjective(self.values.copy())

    def _objective(self, x):
        return -np.matmul(self.values[None, :], x)

    def _gradient(self, x):
        return -np.matmul(self.values[:, None], np.ones((1, x.shape[1]), dtype=float))


def knapsack():
    values = np.asarray([5., 2., 3., 6., 8.], dtype=float)
    weights = np.asarray([[3., 4., 6., 2., 5.]], dtype=float)
    max_weights = np.asarray([17.], dtype=float)

    return sawdown.MipOptimizer().set_objective(KnapsackObjective(values)) \
        .linear_inequality_constraints(-weights, max_weights) \
        .binary_constraints(var_indices=list(range(5))) \
        .steepest_descent().decay_steplength(10) \
        .stop_after(100).stop_small_steps() \
        .config(epsilon=1e-2) \
        .parallelize(4)


def least_square(_a, _b):
    def _objective(_x):
        return 0.5 * np.sum(np.square(np.matmul(_a, _x) - _b[:, None]), axis=0)

    def _deriv(_x):
        return np.matmul(_a.T, np.matmul(_a, _x) - _b[:, None])

    return sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
        .fixed_initializer(np.zeros_like(_a[0, :], dtype=float)) \
        .steepest_descent() \
        .quadratic_interpolation_steplength() \
        .stop_after(100).stop_small_steps()


def simple_inequality_quadratic():
    constraint_a = np.asarray([[1., 0.], [0., 1.]], dtype=float)
    constraint_b = np.asarray([1., 0.], dtype=float)

    # (x + 2)^2 + (y+2)^2 = x^2 + y^2 + 4(x+y)
    obj_func = lambda _: np.sum(np.square(_), axis=0) + 4. * np.sum(_, axis=0)
    deriv_func = lambda _: 2. * _ + 4.

    return sawdown.FirstOrderOptimizer().objective(obj_func, deriv_func) \
        .linear_inequality_constraints(constraint_a, constraint_b) \
        .steepest_descent().quadratic_interpolation_steplength() \
        .stop_after(100).stop_small_steps()

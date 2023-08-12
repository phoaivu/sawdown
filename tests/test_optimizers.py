import glob
import os.path
import unittest
import numpy as np

import sawdown
from tests import plotter
from tests.sawdown import test_problems


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


class OptimizerWrapper(object):
    """
    Hack, until Plotter is fixed for good.
    """

    def __init__(self, new_optimizer):
        self._optimizer = new_optimizer

    def objective(self, data):
        return self._optimizer._objective.objective(data)

    def residuals(self, data):
        residuals = None
        if not self._optimizer._inequality_constraints.is_empty():
            residuals = self._optimizer._inequality_constraints.residuals(data)
        if not self._optimizer._equality_constraints.is_empty():
            equality_residuals = self._optimizer._equality_constraints.residuals(data)
            residuals = equality_residuals if residuals is None else np.vstack((residuals, equality_residuals))
        return residuals


class TestOptimizers(unittest.TestCase):

    def _plot(self, optimizer, iteration_data_reader, to_iter=2):
        if True:
            p = plotter.Plotter(optimizer=OptimizerWrapper(optimizer), reader=iteration_data_reader)
            p.draw(show=True, resolution=100, from_iter=0, to_iter=to_iter)

    @staticmethod
    def _test_diary(log_stream=''):
        return sawdown.diaries.log_stream(log_stream)

    def test_least_square(self):
        solution = test_problems.least_square(np.asarray([[4.]], dtype=float), np.asarray([16], dtype=float)).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([4], dtype=float)))
        self.assertLess(solution.iteration, 2)

        # determined, has an optima.
        a = np.asarray([[2, -3], [-1, -4]], dtype=float)
        b = np.asarray([-6, -16], dtype=float)
        solution = test_problems.least_square(a, b).optimize()
        self.assertEqual(solution.x.shape, (2,))
        self.assertLess(solution.iteration, 29)
        self.assertLess(np.square(solution.objective), 1e-5)

        # two parallel lines
        a = np.asarray([[2, -3], [-2, 3]], dtype=float)
        b = np.asarray([6, -3], dtype=float)
        solution = test_problems.least_square(a, b).optimize()
        self.assertEqual(solution.x.shape, (2,))
        self.assertLess(solution.iteration, 2)
        self.assertLess(solution.objective, 2.3)
        self.assertGreater(solution.objective, 2.2)

        # Under-determined
        a = np.asarray([[2, -3]], dtype=float)
        b = np.asarray([6], dtype=float)
        solution = test_problems.least_square(a, b).optimize()
        self.assertEqual(solution.x.shape, (2,))
        self.assertLess(solution.iteration, 2)
        self.assertLess(np.square(solution.objective), 1e-5)

        # Over-determined linear system.
        a = np.asarray([[2, -3], [-1, -4], [-2, 1]], dtype=float)
        b = np.asarray([6, 16, 4], dtype=float)
        solution = test_problems.least_square(a, b).optimize()
        self.assertEqual(solution.x.shape, (2,))
        self.assertLess(solution.objective, 2.7)
        self.assertGreater(solution.objective, 2.6)
        self.assertLess(solution.iteration, 100)

    def test_simple_inequalities(self):
        # No initialization
        r = test_problems.simple_inequality_quadratic()
        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))

        # Fixed initializer
        r = test_problems.simple_inequality_quadratic()
        solution = r.fixed_initializer(np.asarray([-3., -1.], dtype=float)).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))

        r = test_problems.simple_inequality_quadratic()
        solution = r.fixed_initializer(np.asarray([1., 1.], dtype=float)).stop_after(max_iters=10).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))

        r = test_problems.simple_inequality_quadratic()
        solution = r.fixed_initializer(np.asarray([0., 1.], dtype=float)).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        self._plot(r, solution.iteration_data_reader())

        r = test_problems.simple_inequality_quadratic()
        solution = r.fixed_initializer(np.asarray([20., 21.], dtype=float)).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        self._plot(r, solution.iteration_data_reader())

    def test_toy_inequalities(self):
        # x1 - 2x2 + 5 >= 0
        # 4x1 - x2 + 4 >= 0
        constraint_a = np.asarray([[1., -2.],
                                   [4., -1.]], dtype=float)
        constraint_b = np.asarray([5., 4.], dtype=float)

        def objective(xk):
            return np.sum(np.square(xk), axis=0) + 3. * xk[0, :]

        def deriv(xk):
            return 2. * xk + np.asarray([3., 0.], dtype=float)[:, None]

        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .fixed_initializer(np.asarray([0., 2.5], dtype=float))

        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 2)
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps().config(epsilon=1e-20)

        solution2 = r.optimize()
        self.assertTrue(np.allclose(solution.x, solution2.x, atol=1e-20))
        self.assertEqual(solution2.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertLess(solution2.iteration, 3)
        self._plot(r, solution2.iteration_data_reader(), to_iter=10)

        # x1 - 2x2 + 5 >= 0
        # 4x1 - x2 + 4 >= 0
        # x1 + 3/4 >= 0
        constraint_a = np.asarray([[1., -2.],
                                   [4., -1.],
                                   [1., 0.]], dtype=float)
        constraint_b = np.asarray([5., 4., 3./4], dtype=float)
        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .fixed_initializer(np.asarray([0., 2.5]))
        solution3 = r.optimize()
        self.assertEqual(solution3.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution3.iteration, 3)
        self._plot(r, solution3.iteration_data_reader(), to_iter=10)

    def test_toy_equalities(self):
        def objective(xk):
            return np.sum(np.square(xk), axis=0) + 3. * xk[0, :]

        def deriv(xk):
            return 2. * xk + np.asarray([3., 0.], dtype=float)[:, None]

        # x1 - 2x2 + 3 = 0
        constraint_a = np.asarray([[1., -2.]], dtype=float)
        constraint_b = np.asarray([3.], dtype=float)

        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_equality_constraints(constraint_a, constraint_b) \
            .steepest_descent() \
            .quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 1)
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

        # 2x1 - x2 = 0
        # x1 - 2x2 + 3 = 0
        constraint_a = np.asarray([[2., -1.],
                                   [1., -2.]], dtype=float)
        constraint_b = np.asarray([0., 3.], dtype=float)

        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_equality_constraints(constraint_a, constraint_b) \
            .steepest_descent() \
            .quadratic_interpolation_steplength()\
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 0)
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

    def test_toy_inequalities_equalities(self):
        def objective(xk):
            return np.sum(np.square(xk), axis=0) + 3. * xk[0, :]

        def deriv(xk):
            return 2. * xk + np.asarray([3., 0.], dtype=float)[:, None]

        # 2x1 - x2 = 0
        # x1 - 2x2 + 3 >= 0
        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(np.asarray([[1., -2.]], dtype=float), np.asarray([3.], dtype=float)) \
            .linear_equality_constraints(np.asarray([[2., -1.]], dtype=float), np.asarray([0.], dtype=float)) \
            .steepest_descent() \
            .quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 2)
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

        # 2x1 - x2 - 1 = 0
        # x1 - 2x2 + 3 >= 0
        # x1 - 1/4 >= 0
        constraint_a = np.asarray([[1., -2.],
                                   [1., 0.]], dtype=float)
        constraint_b = np.asarray([3., -0.25], dtype=float)
        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .linear_equality_constraints(np.asarray([[2., -1.]], dtype=float), np.asarray([-1.], dtype=float)) \
            .steepest_descent() \
            .quadratic_interpolation_steplength() \
            .fixed_initializer(np.asarray([-2., 4.], dtype=float)) \
            .stop_after(100).stop_small_steps().log_stream('stdout')
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 1)
        self.assertTrue(np.allclose(solution.x, np.asarray([0.25, -0.5], dtype=float)))
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

        # 2x1 - x2 - 1 >= 0
        # x1 - 2x2 + 3 >= 0
        # x1 = 1/4 (via fixed_value_constraints(), for test only.)
        constraint_a = np.asarray([[2., -1.],
                                   [1., -2.]], dtype=float)
        constraint_b = np.asarray([-1., 3.], dtype=float)
        r = sawdown.FirstOrderOptimizer().objective(objective, deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .fixed_value_constraint(0, 1./4) \
            .steepest_descent() \
            .quadratic_interpolation_steplength() \
            .fixed_initializer(np.asarray([-2., 4.], dtype=float)) \
            .stop_after(100).stop_small_steps().log_stream('stdout')
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 1)
        self.assertTrue(np.allclose(solution.x, np.asarray([0.25, -0.5], dtype=float)))
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

    def test_textbook_inequalities(self):
        # x_1 - x_2 >= 0
        # x_1 + x_2 <= 4
        # x_1 <= 3
        constraint_a = np.asarray([[1., -1.],
                                   [-1, -1.],
                                   [-1, 0.]], dtype=float)
        constraint_b = np.asarray([0., 4., 3.], dtype=float)

        def _deriv(_x):
            # 2 + 8x_1 + 2x_2
            # 3 + 2x_1 + 2x_2
            deriv = (np.matmul(np.asarray([[8., 2.], [2., 2.]], dtype=float), _x)
                     + np.asarray([[2.], [3.]], dtype=float))
            return deriv

        def _objective(_x):
            # 2x_0 + 3x_1 + 4x_0^2 + 2x_0 x_1 + x_1^2
            x0 = _x[0, :]
            x1 = _x[1, :]
            return 2 * x0 + 3 * x1 + 4 * np.square(x0) + 2 * np.prod(_x, axis=0) + np.square(x1)

        initializer = np.asarray([2., 1.])
        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .fixed_initializer(initializer)
        solution = r.optimize()
        self.assertLess(solution.iteration, 50)
        self._plot(r, solution.iteration_data_reader(), to_iter=10)

        initializer = np.asarray([3., 1.])
        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .fixed_initializer(initializer)
        solution2 = r.optimize()
        self.assertLess(solution2.iteration, 30)
        self.assertTrue(np.allclose(solution.x, solution2.x, rtol=1e-5))

        initializer = np.asarray([1., -3.])
        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .fixed_initializer(initializer)
        solution3 = r.optimize()
        self.assertLess(solution3.iteration, 55)
        self.assertTrue(np.allclose(solution.x, solution3.x, rtol=1e-5))

        # Adam
        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .fixed_initializer(initializer=np.asarray([2., 1.], dtype=float)) \
            .adam() \
            .stop_after(500).stop_small_steps()

        solution = r.optimize()
        self.assertIsNotNone(solution.x)
        # self.assertTrue(np.allclose(x, solution.x, rtol=1e-2))
        self._plot(r, solution.iteration_data_reader())

    def test_textbook_inequalities_equalities(self):
        # x_1 - x_2 = 0
        # x_1 + x_2 <= 4
        # x_1 <= 3

        def _deriv(_x):
            # 2 + 8x_1 + 2x_2
            # 3 + 2x_1 + 2x_2
            deriv = (np.matmul(np.asarray([[8., 2.], [2., 2.]], dtype=float), _x)
                     + np.asarray([[2.], [3.]], dtype=float))
            return deriv

        def _objective(_x):
            # 2x_0 + 3x_1 + 4x_0^2 + 2x_0 x_1 + x_1^2
            x0 = _x[0, :]
            x1 = _x[1, :]
            return 2 * x0 + 3 * x1 + 4 * np.square(x0) + 2 * np.prod(_x, axis=0) + np.square(x1)

        constraint_a = np.asarray([[-1., -1.],
                                   [-1., 0.]], dtype=float)
        constraint_b = np.asarray([4., 3.], dtype=float)
        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .linear_equality_constraints(np.asarray([[1., -1.]], dtype=float), np.asarray([0.], dtype=float)) \
            .fixed_initializer(np.asarray([5., 2.5], dtype=float)) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-0.35714286, -0.35714286], dtype=float)))
        self.assertEqual(solution.iteration, 1)
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self._plot(r, solution.iteration_data_reader())

    def test_3d_inequalities(self):
        # x + z + 8 >= 0
        # x <= 3
        # x + 2y -z + 6 >= 0
        constraint_a = np.asarray([[1., 0., 1.],
                                   [-1., 0., 0.],
                                   [1., 2., -1.]], dtype=float)
        constraint_b = np.asarray([8., 3., 6.], dtype=float)

        def _objective(x):
            # (x+2)^2 + (3y-2)^2 + (z+1)^2
            return np.square(x[0, :] + 2.) + np.square((3. * x[1, :]) - 2.) - np.square(x[2, :] + 1.)

        def _deriv(x):
            # 2x + 4; 18y - 12; 2z + 2
            return (np.multiply(np.asarray([2., 18., 2.], dtype=float)[:, None], x)
                    + np.asarray([4., -12., 2.], dtype=float)[:, None])

        r = sawdown.FirstOrderOptimizer().objective(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-2., 2. / 3, -1.], dtype=float)))

        solution = r.fixed_initializer(np.asarray([3., -4.5, 0.], dtype=float)).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-2., 2. / 3, -1.], dtype=float)))

        '''
        import scipy.optimize
        constraints = [scipy.optimize.LinearConstraint(constraint_a, -constraint_b,
                                                       ub=np.inf*np.ones_like(constraint_b), keep_feasible=True)]
        solution = scipy.optimize.minimize(lambda _x: _objective(_x[:, None]).squeeze(),
                                           np.asarray([3., -4.5, 0.], dtype=float),
                                           method='trust-constr',
                                           jac=lambda _x: _deriv(x[:, None]).squeeze(),
                                           constraints=constraints,
                                           options=dict(maxiter=1000, disp=True))
        print(solution.x)
        print(solution.success)
        print(solution.message)
        '''

    def test_unconstrained_mip(self):
        # min 0.5*||(1.4 - x, 1.6 - y)||^2
        def _objective(x):
            return 0.5 * np.sum(np.square(np.asarray([1.4, 1.6], dtype=float)[:, None] - x), axis=0)
        
        def _grad(x):
            return x - np.asarray([1.4, 1.6], dtype=float)[:, None]

        r = sawdown.MipOptimizer().objective(_objective, _grad) \
            .integer_constraint(var_index=0) \
            .integer_constraint(var_index=1) \
            .fixed_initializer(np.zeros((2,), dtype=float)) \
            .steepest_descent() \
            .quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .parallelize(0) \
            .log_stream('stdout')
        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 2.])))
        self.assertLess(solution.iteration, 7)

    def test_constrained_mip(self):
        # min -0.5*(y - 100)^2
        # x - y + 1 >= 0
        # -3x -2y + 12 >= 0
        # -2x -3y + 12 >= 0
        # x >= 0
        # y >= 0
        # x, y integers.
        constraint_a = np.asarray([[1, -1], [-3, -2], [-2, -3]], dtype=float)
        constraint_b = np.asarray([1, 12, 12], dtype=float)

        r = sawdown.MipOptimizer().objective(
            objective_func=lambda _x: -0.5 * np.square(_x[1, :] - 100.),
            deriv_func=lambda _x: np.vstack((np.zeros_like(_x[0, :], dtype=float), (_x[1, :] - 100)))) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .integer_constraint(var_index=0, lower_bound=0.) \
            .integer_constraint(var_index=1, lower_bound=0.) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps() \
            .parallelize(0) \
            .log_stream('stdout')

        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 2.], dtype=float)))
        self.assertLess(solution.iteration, 5)

        # min -y
        r = sawdown.MipOptimizer().objective(
            objective_func=lambda _x: -_x[1, :],
            deriv_func=lambda _x: np.zeros_like(_x) + np.asarray([0., -1.], dtype=float)[:, None]) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .integer_constraint(var_index=0, lower_bound=0.) \
            .integer_constraint(var_index=1, lower_bound=0.) \
            .steepest_descent().decay_steplength(10) \
            .stop_after(100).stop_small_steps() \
            .parallelize(0) \
            .log_stream('stdout')
        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 2.], dtype=float)))
        self.assertLess(solution.iteration, 9)

    def test_knapsack(self):
        log_path = os.path.join(os.path.split(__file__)[0], 'logs')
        for folder in glob.glob(os.path.join(log_path, 'knapsack*')):
            [os.remove(f) for f in glob.glob(os.path.join(folder, '*.*'))]
            os.rmdir(folder)

        r = test_problems.knapsack().log_file(log_path, 'knapsack')

        solution = r.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertLess(solution.iteration, 15)
        self.assertEqual(solution.objective, -22.)
        # iteration data is readable with a log_file()
        self.assertTrue(os.path.exists(os.path.join(log_path, 'knapsack', 'knapsack.pkl')))
        reader = sawdown.diaries.MemoryReader(pickle_file=os.path.join(log_path, 'knapsack', 'knapsack.pkl'))
        reread_solution = reader.solution()
        self.assertEqual(solution['spent_time'], reread_solution['spent_time'])

    def test_circular_steps(self):
        # Keeps alternating between (0, 1) and (2, 1)
        def _objective(x):
            return 100. * (np.square(x[0, :] - 1.) + np.square(x[1, :] + 1.))

        def _grad(x):
            return 200. * (x + np.asarray([-1., 1.], dtype=float)[:, None])

        # Equality constraint: x[1] = 1. The minimum is 400 at (1, 1)
        r = sawdown.FirstOrderOptimizer().objective(_objective, _grad) \
            .linear_equality_constraints(np.asarray([[0., 1.]], dtype=float),
                                         np.asarray([-1.], dtype=float)) \
            .linear_inequality_constraints(np.asarray([[1., 0.], [-1., 0.]], dtype=float),
                                           np.asarray([0., 2.], dtype=float)) \
            .steepest_descent().decay_steplength(40) \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.MAX_ITERATION)
        self.assertGreater(solution.objective, 499.)
        self.assertFalse(np.allclose(solution.x, np.asarray([1., 1.], dtype=float)))

        # Quadratic interpolation steplength
        solution = r.quadratic_interpolation_steplength().optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 1)
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 1.], dtype=float)))

        # Adam
        r = sawdown.FirstOrderOptimizer().objective(_objective, _grad) \
            .linear_equality_constraints(np.asarray([[0., 1.]], dtype=float),
                                         np.asarray([-1.], dtype=float)) \
            .linear_inequality_constraints(np.asarray([[1., 0.], [-1., 0.]], dtype=float),
                                           np.asarray([0., 2.], dtype=float)) \
            .adam(alpha=0.95, beta=0.999) \
            .stop_after(100).stop_small_steps()
        solution = r.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.MAX_ITERATION)
        self.assertLess(solution.objective, 402.)


if __name__ == '__main__':
    unittest.main()

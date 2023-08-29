import unittest
import numpy as np

import sawdown
import sawdown as sd


class TestSymbolic(unittest.TestCase):

    def test_easy(self):
        def _objective(x, y):
            return 0.5 * sd.ops.sum(sd.ops.square(x + y))

        optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(5, 1)) \
            .fixed_initializer(np.ones((6,), dtype=float)) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()

        solution = optimizer.optimize()
        self.assertEqual(solution.termination, sawdown.Termination.INFINITESIMAL_STEP)
        self.assertEqual(solution.iteration, 1)
        self.assertTrue(np.allclose(solution.x, np.asarray([2./3, 2./3, 2./3, 2./3, 2./3, -2./3], dtype=float)))

    def test_least_square_with_l2_norm(self):
        a = np.arange(15, dtype=float).reshape((3, 5))
        b = np.ones((3,), dtype=float)

        def _objective(x):
            x_cost = sd.ops.matmul(a, x) + (-b[:, None])
            x_cost = sd.ops.sum(sd.ops.square(x_cost), axis=0)
            reg_cost = 0.1 * sd.ops.sum(sd.ops.square(x))
            return (0.5 * x_cost) + reg_cost

        optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(5,)) \
            .fixed_initializer(np.zeros((5, ), dtype=float)) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps().epsilon(1e-5)

        solution = optimizer.optimize()
        self.assertEqual(solution.iteration, 6)
        self.assertTrue(np.allclose(solution.x,
                                    np.asarray([-0.19165316, -0.09542342,  0.00080632,  0.09703605,  0.19326579],
                                               dtype=float), atol=1e-5))

    def test_multiple_arguments(self):
        a = np.arange(15).reshape((3, 5))
        b = np.ones((3,), dtype=float)

        def _objective(x, y):
            x_cost = sd.ops.matmul(a, x) + b[:, None]
            x_cost = sd.ops.sum(sd.ops.square(x_cost))
            y_cost = 0.2 * sd.ops.sum(sd.ops.square(y))
            return x_cost + y_cost

        optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(5, 4)) \
            .fixed_initializer(np.ones((9, ), dtype=float)) \
            .steepest_descent().decayed_steplength(decay_steps=20, initial_steplength=1e-3) \
            .stop_after(100).stop_small_steps().epsilon(1e-5)

        solution = optimizer.optimize()
        self.assertEqual(solution.iteration, 12)
        self.assertTrue(np.allclose(solution.x,
                                    np.asarray([0.2138405 ,  0.10758386,  0.00132722, -0.10492942, -0.21118606,
                                                0.99608704,  0.99608704,  0.99608704,  0.99608704], dtype=float),
                                    atol=1e-5))


if __name__ == '__main__':
    unittest.main()

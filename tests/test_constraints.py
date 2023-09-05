import unittest
import numpy as np

import sawdown
from sawdown.constraints import inequalities
from sawdown.constraints import equalities


class TestConstraints(unittest.TestCase):

    def test_linear_inequalities_initializer(self):

        from sawdown import opti_math, config
        _opti_math = opti_math.OptiMath()
        _config = config.Config()

        a = np.asarray([[1., -1], [-2., -3.], [-3., -2.]], dtype=float)
        b = np.asarray([1., 12., 12.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)

        initializer = None
        with sawdown.test_diary() as diary:
            x = r.initialize(initializer, _config, _opti_math, diary)
        self.assertTrue(np.allclose(diary.solution.x, x))
        self.assertTrue(np.allclose(diary.solution.x, np.asarray([0., 0.], dtype=float)))
        self.assertEqual(diary.solution.termination, sawdown.Termination.SATISFIED)

        initializer = np.asarray([4., 4.], dtype=float)
        with sawdown.test_diary() as diary:
            r.initialize(initializer, _config, _opti_math, diary)
        self.assertEqual(diary.solution.termination, sawdown.Termination.SATISFIED)
        self.assertEqual(diary.solution.iteration, 1)

        # infeasible case
        a = np.asarray([[1., -1], [-2., -3.], [-3., -2.], [0., 1.]], dtype=float)
        b = np.asarray([1., 12., 12., -3.], dtype=float)

        initializer = None
        r = inequalities.LinearInequalityConstraints(a, b)
        with sawdown.test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, _config, _opti_math, diary)
        self.assertEqual(diary.solution.termination, sawdown.Termination.MAX_ITERATION)

        initializer = np.asarray([4., 4.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)
        with sawdown.test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, _config, _opti_math, diary)
        self.assertEqual(diary.solution.termination, sawdown.Termination.MAX_ITERATION)

        # parallel lines: feasible
        a = np.asarray([[1., -1], [-1., 1.]], dtype=float)
        b = np.asarray([1., 0.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)

        initializer = np.asarray([0, 3.], dtype=float)
        with sawdown.test_diary() as diary:
            r.initialize(initializer, _config, _opti_math, diary)
        self.assertEqual(diary.solution.termination, sawdown.Termination.SATISFIED)

        # parallel lines: infeasible
        a = np.asarray([[-1., 1], [1., -1.]], dtype=float)
        b = np.asarray([-1., 0.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)

        initializer = None
        with sawdown.test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, _config, _opti_math, diary)
        self.assertEqual(diary.solution.termination, sawdown.Termination.MAX_ITERATION)

    def test_linear_inequality_constraints(self):
        # x_1 + 2x_2 >= 1
        # x_1 - x_2 >= -1
        # x_1 <= 3
        constraint_a = np.asarray([[1., 2.],
                                   [1., -1.],
                                   [-1., 0]], dtype=float)
        constraint_b = np.asarray([-1., 1., 3.], dtype=float)
        c = inequalities.LinearInequalityConstraints(constraint_a, constraint_b)

        x = np.asarray([2., 1])
        self.assertEqual(c.residuals(x).shape, (3,))
        self.assertEqual(c._projectors.shape, (3, 2, 2))

        x = np.asarray([[2., 3., -1.],
                        [1., -2., 0.]], dtype=float)
        self.assertEqual(c.residuals(x).shape, (3, 3))
        self.assertEqual(c._projectors.shape, (3, 2, 2))

    def test_least_square(self):
        a = np.asarray([[1., 1., 1., 1., 1.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.,
                         1., 1., 1., 1., 1.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         1., 1., 1., 1., 1.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         1., 1., 1., 1., 1.,
                         0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0.,
                         1., 1., 1., 1., 1.],
                        [1., 0., 0., 0., 0.,
                         1., 0., 0., 0., 0.,
                         1., 0., 0., 0., 0.,
                         1., 0., 0., 0., 0.,
                         1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.,
                         0., 1., 0., 0., 0.,
                         0., 1., 0., 0., 0.,
                         0., 1., 0., 0., 0.,
                         0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 1., 0., 0.,
                         0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.,
                         0., 0., 0., 1., 0.,
                         0., 0., 0., 1., 0.,
                         0., 0., 0., 1., 0.,
                         0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 1.,
                         0., 0., 0., 0., 1.],
                        [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.],
                        [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                         0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float)
        b = np.asarray([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -0., -0., -0., -0.], dtype=float)

        from tests import test_problems
        from sawdown import opti_math, config, diaries

        optimizer = sawdown.FirstOrderOptimizer().objective_instance(test_problems.LeastSquareObjective, a, -b) \
            .fixed_initializer(np.zeros((a.shape[1],), dtype=float)) \
            .bound_constraints(range(a.shape[1]), lower_bounds=0., upper_bounds=1.) \
            .epsilon(1e-10) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(500).stop_small_steps()
        solution = optimizer.optimize()
        residuals = np.matmul(a, solution.x[:, None]) + b[:, None]
        print(solution)
        self.assertTrue(np.all(opti_math.SawMath(epsilon=1e-10).equal_zeros(residuals)))

        c = equalities.LinearEqualityConstraints(a, b)
        m = opti_math.OptiMath(epsilon=1e-10)
        with diaries.SyncDiary('0', [], None) as diary:
            initialized = c.initialize(np.zeros((a.shape[1], )), config.Config(), m, diary)
        self.assertTrue(c.satisfied(initialized, m))

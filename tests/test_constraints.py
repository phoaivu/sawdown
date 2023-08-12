import unittest
import numpy as np

import sawdown
from sawdown.constraints import inequalities


class TestConstraints(unittest.TestCase):

    def _test_diary(self):
        return sawdown.diaries.log_stream('')

    def test_linear_initializer(self):

        from sawdown import opti_math

        a = np.asarray([[1., -1], [-2., -3.], [-3., -2.]], dtype=float)
        b = np.asarray([1., 12., 12.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)
        r.setup(objective=None, opti_math=opti_math.OptiMath())

        initializer = None
        with self._test_diary() as diary:
            x = r.initialize(initializer, diary)
            self.assertTrue(np.allclose(diary.solution.x, x))
            self.assertTrue(np.allclose(diary.solution.x, np.asarray([0., 0.], dtype=float)))
            self.assertEqual(diary.solution.termination, sawdown.Termination.SATISFIED)

        initializer = np.asarray([4., 4.], dtype=float)
        with self._test_diary() as diary:
            r.initialize(initializer, diary)
            self.assertEqual(diary.solution.termination, sawdown.Termination.SATISFIED)
            self.assertEqual(diary.solution.iteration, 1)

        # infeasible case
        a = np.asarray([[1., -1], [-2., -3.], [-3., -2.], [0., 1.]], dtype=float)
        b = np.asarray([1., 12., 12., -3.], dtype=float)

        initializer = None
        r = inequalities.LinearInequalityConstraints(a, b)
        r.setup(objective=None, opti_math=opti_math.OptiMath())
        with self._test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, diary)
            self.assertEqual(diary.solution.termination, sawdown.Termination.MAX_ITERATION)

        initializer = np.asarray([4., 4.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)
        r.setup(objective=None, opti_math=opti_math.OptiMath())
        with self._test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, diary)
            self.assertEqual(diary.solution.termination, sawdown.Termination.MAX_ITERATION)

        # parallel lines: feasible
        a = np.asarray([[1., -1], [-1., 1.]], dtype=float)
        b = np.asarray([1., 0.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)
        r.setup(objective=None, opti_math=opti_math.OptiMath())

        initializer = np.asarray([0, 3.], dtype=float)
        with self._test_diary() as diary:
            r.initialize(initializer, diary)

        # parallel lines: infeasible
        a = np.asarray([[-1., 1], [1., -1.]], dtype=float)
        b = np.asarray([-1., 0.], dtype=float)
        r = inequalities.LinearInequalityConstraints(a, b)
        r.setup(objective=None, opti_math=opti_math.OptiMath())

        initializer = None
        with self._test_diary() as diary:
            with self.assertRaises(sawdown.InitializationError):
                r.initialize(initializer, diary)
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

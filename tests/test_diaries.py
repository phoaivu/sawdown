import os.path
import glob
import unittest
import numpy as np

import sawdown
from tests import test_problems


class TestDiaries(unittest.TestCase):

    @staticmethod
    def get_optimizer():
        constraint_a = np.asarray([[1., 0.], [0., 1.]], dtype=float)
        constraint_b = np.asarray([1., 0.], dtype=float)

        # (x + 2)^2 + (y+2)^2 = x^2 + y^2 + 4(x+y)
        def _objective(_x):
            return np.sum(np.square(_x), axis=0) + 4. * np.sum(_x, axis=0)

        def _deriv(_x):
            return 2. * _x + 4.

        return sawdown.FirstOrderOptimizer().objective_functors(_objective, _deriv) \
            .linear_inequality_constraints(constraint_a, constraint_b) \
            .fixed_initializer(np.asarray([-3., -1.])) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps()

    @classmethod
    def setUpClass(cls):
        for path in glob.glob(os.path.join(os.path.split(__file__)[0], 'logs', 'test_diaries_*')):
            [os.remove(p) for p in glob.glob(os.path.join(path, '*'))]
            os.rmdir(path)

    def test_stream(self):
        optimizer = TestDiaries.get_optimizer()

        # Default config: returned solution, no iteration data
        solution = optimizer.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        with self.assertRaises(RuntimeError):
            solution.iteration_data_reader()

        # Minimal config to return iteration data.
        optimizer = TestDiaries.get_optimizer().diary()
        solution = optimizer.optimize()
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, ('iteration', 'x_k', 'd_k')))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

        # Write to stdout, should be able to read data from the reader
        optimizer = TestDiaries.get_optimizer().diary().stream_diary('stdout')
        solution = optimizer.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, ('iteration', 'x_k', 'd_k')))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

    def test_file_writer(self):
        optimizer = TestDiaries.get_optimizer()

        optimizer.file_diary(os.path.join(os.path.split(__file__)[0], 'logs'), job_name='test_diaries_sync')
        solution = optimizer.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, ('iteration', 'x_k', 'd_k')))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

    def test_mip_knapsack(self):
        optimizer = test_problems.knapsack()

        # Default configuration (i.e. no iteration data), with parallelization
        solution = optimizer.parallelize(4).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        with self.assertRaises(RuntimeError):
            solution.iteration_data_reader()
        print(solution['spent_time'])

        # With iteration data, with parallelization
        solution = test_problems.knapsack().parallelize(4).diary().optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        print(solution['spent_time'])

        # With iteration data, without parallelization
        solution = test_problems.knapsack().parallelize(0).diary().optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        print(solution['spent_time'])

        # File diary
        log_path = os.path.join(os.path.split(__file__)[0], 'logs')
        solution = test_problems.knapsack().parallelize(4).file_diary(log_path, 'test_diaries_knapsack').optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        print(solution['spent_time'])

    def test_mip_gracefully_terminate(self):
        optimizer = test_problems.knapsack_fault().stream_diary('stdout')
        with self.assertRaises(ValueError):
            optimizer.optimize()


if __name__ == '__main__':
    unittest.main()

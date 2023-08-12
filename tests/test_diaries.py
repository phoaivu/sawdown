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
        obj_func = lambda _: np.sum(np.square(_), axis=0) + 4. * np.sum(_, axis=0)
        deriv_func = lambda _: 2. * _ + 4.

        return sawdown.FirstOrderOptimizer().objective(obj_func, deriv_func) \
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

        # Default config: StreamWriter without writing anything.
        solution = optimizer.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, 'iteration', 'x_k', 'd_k'))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

        # Write to stdout, should be able to read data from the reader
        with sawdown.diaries.log_stream('stdout') as diary:
            solution = optimizer.optimize(diary)
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, 'iteration', 'x_k', 'd_k'))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

    def test_file_writer(self):
        optimizer = TestDiaries.get_optimizer()

        optimizer.log_file(os.path.join(os.path.split(__file__)[0], 'logs'), job_name='test_diaries_sync')
        solution = optimizer.optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([-1., 0.])))
        self.assertEqual(solution.iteration, 2)
        reader = solution.iteration_data_reader()
        data = list(reader.iterations(None, 'iteration', 'x_k', 'd_k'))
        self.assertEqual(len(data), 3)
        self.assertEqual(reader.solution(None).iteration, 2)

    def test_mip_knapsack(self):
        optimizer = test_problems.knapsack()

        # Default diary, without parallelization
        # solution = optimizer.parallelize(0).optimize()
        # self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        # self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        # print(solution['spent_time'])

        # # Default diary, with parallelization
        solution = optimizer.parallelize(4).optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        print(solution['spent_time'])

        # File diary
        log_path = os.path.join(os.path.split(__file__)[0], 'logs')
        solution = optimizer.log_file(log_path, 'test_diaries_knapsack').optimize()
        self.assertTrue(np.allclose(solution.x, np.asarray([1., 0., 1., 1., 1.], dtype=float)))
        self.assertIsNotNone(solution.iteration_data_reader().iterations(None))
        print(solution['spent_time'])


if __name__ == '__main__':
    unittest.main()

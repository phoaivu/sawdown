import unittest
import numpy as np

import sawdown as sd

class TestSymbolic(unittest.TestCase):

    def test_quadratic(self):

        a = np.arange(15).reshape((3, 5))
        b = np.ones((3,), dtype=float)

        def _objective(x, y):
            x_cost = sd.ops.matmul(a, x) + b[:, None]
            x_cost = sd.ops.sum(sd.ops.square(x_cost))
            y_cost = 0.2 * sd.ops.sum(sd.ops.square(y))
            return x_cost + y_cost

        optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(5, 4)) \
            .fixed_initializer(np.ones((9, ), dtype=float)) \
            .adam() \
            .stop_after(100).stop_small_steps().stream_diary('stdout')

        solution = optimizer.optimize()
        print(solution)


if __name__ == '__main__':
    unittest.main()

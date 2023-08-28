import unittest
import numpy as np

import sawdown as sd


class TestSymbolic(unittest.TestCase):

    def test_easy(self):
        def _objective(x, y):
            return (0.5 * sd.ops.square(x + 2.)) + sd.ops.square(y + 4.)

        from sawdown import tensorcube as tc
        with tc.Graph() as g:
            x = tc.Variable('x', expected_shape=(1, 1))
            y = tc.Variable('y', expected_shape=(1, 1))
            obj = _objective(x, y)

            x.set_value(np.ones((1, 1), dtype=float))
            y.set_value(np.ones((1, 1), dtype=float))
            print(obj.evaluate())
            grad_x, grad_y = obj.gradients((x, y))
            print(grad_x.evaluate())
            print(grad_y.evaluate())

            x.set_value(2 * np.ones((1, 1), dtype=float))
            y.set_value(3 * np.ones((1, 1), dtype=float))
            print(obj.evaluate())
            # grad_x, grad_y = obj.gradients((x, y))
            print(grad_x.evaluate())
            print(grad_y.evaluate())

        proto = g.to_proto()
        with tc.read_graph(proto) as g2:
            x = g2.get_node(x.name)
            y = g2.get_node(y.name)
            obj = g2.get_node(obj.name)
            grad_x = g2.get_node(grad_x.name)
            grad_y = g2.get_node(grad_y.name)

            x.set_value(4 * np.ones((1, 1), dtype=float))
            y.set_value(5 * np.ones((1, 1), dtype=float))
            print(obj.evaluate())
            # grad_x, grad_y = obj.gradients((x, y))
            print(grad_x.evaluate())
            print(grad_y.evaluate())

        return

        optimizer = sd.FirstOrderOptimizer().objective_symbolic(_objective, var_dims=(1, 1)) \
            .fixed_initializer(np.ones((2,), dtype=float)) \
            .steepest_descent().quadratic_interpolation_steplength() \
            .stop_after(100).stop_small_steps().stream_diary('stdout')

        solution = optimizer.optimize()
        print(solution)

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

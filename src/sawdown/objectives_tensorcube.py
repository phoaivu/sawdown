import numpy as np
from sawdown import objectives
from sawdown import tensorcube as tc
from sawdown.proto import sawdown_pb2


def tensorcube_objective(func, var_dims=()):
    """
    :param func: objective function taking tensorcube's tensors as input.
    :param var_dims: int or tuple of ints, the dimensions of variables getting passed into func()
    :rtype: TensorcubeObjective
    """
    if isinstance(var_dims, int):
        var_dims = (var_dims, )
    if not isinstance(var_dims, tuple):
        raise ValueError('var_dims expected to be an int or tuple of int.')
    # These checks could be further improved.
    if len(var_dims) != func.__code__.co_argcount:
        raise ValueError('Function requires {} arguments'.format(func.__code__.co_argcount))

    with tc.Graph() as g:
        variables = tuple(tc.Variable('x{}'.format(i), expected_shape=(var_dims[i], 1)) for i in range(len(var_dims)))
        obj = func(*variables)

        for d, v in zip(var_dims, variables):
            v.set_value(np.zeros((d, 1), dtype=float))
        evaluated_obj = obj.evaluate()
        if evaluated_obj.size != 1:
            raise ValueError('Objective value is not a scalar')
        grads = obj.gradients(variables)

    return TensorcubeObjective(g, [v.name for v in variables], var_dims, obj.name, [g.name for g in grads])



class TensorcubeObjective(objectives.ObjectiveBase):

    def __init__(self, graph, var_names=(), var_dims=(), objective_name='', grad_names=()):
        self._graph = graph
        self._var_names = tuple(var_names)
        self._var_dims = tuple(var_dims)
        self._objective_name = objective_name
        self._grad_names = tuple(grad_names)

        self._vars = [self._graph.get_node(n) for n in self._var_names]
        self._obj = self._graph.get_node(objective_name)
        self._grads = [self._graph.get_node(n) for n in self._grad_names]

    def _objective(self, x):
        assert x.shape == (sum(self._var_dims), 1)
        with self._graph:
            idx = 0
            for v, d in zip(self._vars, self._var_dims):
                v.set_value(x[idx:idx+d, :])
                idx += d
            return self._obj.evaluate()

    def _gradient(self, x):
        assert x.shape == (sum(self._var_dims), 1)
        with self._graph:
            idx = 0
            for v, d in zip(self._vars, self._var_dims):
                v.set_value(x[idx:idx + d, :])
                idx += d
            self._obj.evaluate()
            grads = [g.evaluate() for g in self._grads]
        x_grads = np.zeros_like(x, dtype=float)
        idx = 0
        for g, d in zip(grads, self._var_dims):
            x_grads[idx:idx+d, :] = g
        return x_grads

    def check_dimensions(self, var_dim):
        # checking signature and dimensions of the objective function and derivative computation.
        if self.objective(np.zeros((var_dim,), dtype=float)).size != 1:
            raise ValueError('Objective function must take a {}x1 matrix and return a 1x1 matrix'.format(var_dim))

        if self.deriv_variables(np.zeros((var_dim,), dtype=float)).shape != (var_dim,):
            raise ValueError('Derivative w.r.t. variables takes a {}x1 matrix and return a {}x1 matrix'.format(
                var_dim, var_dim))

    def to_proto(self):
        proto_obj = sawdown_pb2.TensorcubeObjective(graph=self._graph.to_proto(), objective=self._objective_name)
        for d, var_name, grad_name in zip(self._var_dims, self._var_names, self._grad_names):
            proto_obj.variables.append(var_name)
            proto_obj.var_dims.append(d)
            proto_obj.gradients.append(grad_name)
        return proto_obj

    @staticmethod
    def from_proto(proto):
        g = tc.read_graph(proto.graph)
        return TensorcubeObjective(g, proto.variables, proto.var_dims, proto.objective, proto.gradients)

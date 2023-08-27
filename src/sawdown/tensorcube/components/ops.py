import numpy as np

from sawdown.tensorcube.components import nodes_tensor


def ones(shape=(), name=''):
    return nodes_tensor.Constant(np.ones(shape, dtype=float), name=name)


def constant(value, name=''):
    return nodes_tensor.Constant(value, name=name)


def zeros_like(x, name=''):
    return nodes_tensor.ZerosLike(x, name=name)


def transpose(x, axes=None, name=''):
    return nodes_tensor.Transpose(x, axes, name=name)


def square(x, name=''):
    return nodes_tensor.Square(x, name=name)


def sum(x, axis=None, keepdims=False, name=''):
    return nodes_tensor.Sum(x, axis, keepdims, name=name)


def add(x1, x2, name=''):
    return nodes_tensor.Add(x1, x2, name=name)


def multiply(x1, x2, name=''):
    return nodes_tensor.Multiply(x1, x2, name=name)


def matmul(x1, x2, name=''):
    return nodes_tensor.Matmul(x1, x2, name=name)


def tensordot(x1, x2, axes=2, name=''):
    return nodes_tensor.TensorDot(x1, x2, axes, name=name)


def sum_list(*tensors, name=''):
    return nodes_tensor.SumList(*tensors, name=name)



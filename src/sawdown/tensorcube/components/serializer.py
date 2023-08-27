import numpy as np
from sawdown.proto import tensorcube_pb2


def to_ndarray(arr):
    return tensorcube_pb2.NdArray(shape=arr.shape, values=arr.flatten(order='C').tolist(), dtype=str(arr.dtype))


def from_ndarray(proto):
    return np.asarray(proto.values, dtype=proto.dtype).reshape(proto.shape, order='C')

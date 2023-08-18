import importlib
import pickle
import types

import numpy as np
from sawdown.proto import sawdown_pb2

# Serialize and deserialize python objects into/from protobuf messages.


def encode_ndarray(arr):
    return sawdown_pb2.NdArray(shape=arr.shape, values=arr.flatten(order='C').tolist(), dtype=str(arr.dtype))


def decode_ndarray(proto_msg):
    return np.array(proto_msg.values, dtype=proto_msg.dtype).reshape(proto_msg.shape, order='C')


def encode_args(*args):
    vals = []
    mappers = [
        (bool, lambda x: sawdown_pb2.Value(bool_value=x)),
        (str, lambda x: sawdown_pb2.Value(string_value=x)),
        (int, lambda x: sawdown_pb2.Value(int_value=x)),
        (float, lambda x: sawdown_pb2.Value(float_value=x)),
        (bytes, lambda x: sawdown_pb2.Value(bytes_value=x)),
        (np.ndarray, lambda x: sawdown_pb2.Value(array_value=encode_ndarray(x)))
    ]
    for arg in args:
        mapper = next((m for t, m in mappers if isinstance(arg, t)), None)
        if mapper is None:
            vals.append(sawdown_pb2.Value(binary_value=pickle.dumps(arg)))
        else:
            vals.append(mapper(arg))
    return vals


def decode_args(proto_msg):
    vals = []
    for value in proto_msg:
        field_name = value.WhichOneof('value')
        if field_name in {'bool_value', 'string_value', 'int_value', 'float_value', 'bytes_value'}:
            vals.append(getattr(value, field_name))
        elif field_name == 'array_value':
            vals.append(decode_ndarray(value.array_value))
        elif field_name == 'binary_value':
            vals.append(pickle.loads(value.binary_value))
    return vals


def encode_method(method, *args):
    return sawdown_pb2.Method(name=method.__name__, module=method.__module__,
                              args=encode_args(*args))


def decode_method(proto_msg):
    args = decode_args(proto_msg.args)
    return getattr(importlib.import_module(proto_msg.module), proto_msg.name)(*args)


def encode_functor(func):
    """
    Rather hackish to include locally-declared and probably lambda functions.
    """
    code_obj = func.__code__
    data = ((code_obj.co_argcount, code_obj.co_kwonlyargcount, code_obj.co_nlocals, code_obj.co_stacksize,
             code_obj.co_flags, code_obj.co_code, code_obj.co_consts, code_obj.co_names, code_obj.co_varnames,
             code_obj.co_filename, code_obj.co_name, code_obj.co_firstlineno, code_obj.co_lnotab),
            None, func.__closure__)
    return pickle.dumps(data)


def decode_functor(data):
    data = pickle.loads(data)
    return types.FunctionType(types.CodeType(*data[0]), globals(), closure=data[2])

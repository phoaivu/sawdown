import itertools

import numpy as np

from sawdown.tensorcube.components import graph_stack, nodes, ops, serializer
from sawdown.proto import tensorcube_pb2


# TODO: _compute_backward() of all nodes use the evaluated shape of the inputs. (i.e. `.value.shape`)
#  That should be fixed to use the compile-time partial shape `.expected_shape` instead.

class TensorNode(nodes.Node):

    def __init__(self, name='', expected_shape=None):
        """

        :param name:
        :param expected_shape: A list or tuple of ints, with Nones indicating unspecified at compile time.
        """
        nodes.Node.__init__(self, name)
        self._expected_shape = None if expected_shape is None else tuple(expected_shape)
        self._shape = nodes.ShapeNode(self.name + '_shape')

        # The tensorNode containing gradients of a certain descendant node (when gradients() is called).
        self._gradient = None

    @property
    def shape(self):
        return self._shape

    @property
    def expected_shape(self):
        """
        The compile-time deducible shape, if available. None otherwise.
        :return:
        """
        return self._expected_shape

    @property
    def gradient(self):
        return self._gradient

    def set_gradient(self, grad_node):
        """
        Only used in deserialization
        :type grad_node: tensorcube.components.nodes_tensor.TensorNode
        """
        self._gradient = grad_node

    def to_proto(self):
        node_data = nodes.Node.to_proto(self)
        if self._expected_shape is not None:
            for d in self._expected_shape:
                node_data.expected_shape.append(-1 if d is None else d)
        if self._gradient is not None:
            node_data.gradient_name = self._gradient.name
        proto_cls = getattr(tensorcube_pb2, self.__class__.__name__)
        return proto_cls(node_data=node_data)

    def _verify_shape(self, value):
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if self._expected_shape is not None:
            if len(value.shape) != len(self._expected_shape):
                raise RuntimeError('Invalid shape in {}: expected {}, got {}'.format(
                    self.name, self._expected_shape, value.shape))
            if any(e is not None and e != v for e, v in zip(self._expected_shape, value.shape)):
                raise RuntimeError('Invalid shape in {}: expected {}, got {}'.format(
                    self.name, self._expected_shape, value.shape))
        self._shape.set_value(value.shape)

    def _evaluate(self):
        self._compute()
        self._verify_shape(self._value)

    def gradients(self, ancestors):
        """
        Return a (tuple of) Node that would compute the gradients of self` w.r.t. the set of nodes in ancestors.
        :type ancestors: Union[Node, tuple]
        :return:
        """
        tuple_input = isinstance(ancestors, (list, tuple))
        ancestors = tuple(ancestors) if tuple_input else (ancestors,)
        path = graph_stack.active_graph().traverse(self, ancestors)[::-1]

        node, _ = path[0]
        assert node == self
        self._gradient = ops.ones(shape=self.shape.value)
        self._compute_backward(self._gradient)

        for node, outgoing_slots in path[1:]:
            node._backward(outgoing_slots)

        grads = tuple(n.gradient for n in ancestors)
        return grads if tuple_input else grads[0]

    def _backward(self, outgoing_slots):
        children = []
        for slot in outgoing_slots:
            if slot.gradient is not None:
                children.append(slot.gradient)
        if len(children) == 0:
            raise RuntimeError('Node {} does not have any accumulated gradients'.format(self.name))
        if len(children) == 1:
            self._gradient = ops.multiply(children[0], 1.)
        elif len(children) == 2:
            self._gradient = ops.add(children[0], children[1])
        else:
            self._gradient = ops.sum_list(*children)
        self._compute_backward(self._gradient)

    def _compute(self):
        raise NotImplementedError()

    def _compute_backward(self, accumulated_grads):
        """
        From the gradient in self._gradient, and possibly self.value,
        construct the operations to store into slot.gradient.
        :param accumulated_grads:
        :return:
        """
        raise NotImplementedError()

    def __add__(self, other):
        return ops.add(self, other)

    def __radd__(self, other):
        return ops.add(other, self)

    def __mul__(self, other):
        return ops.multiply(self, other)

    def __rmul__(self, other):
        return ops.multiply(other, self)


class Constant(TensorNode):
    def __init__(self, value, name=''):
        if value is None:
            raise ValueError('None value')
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        TensorNode.__init__(self, name, expected_shape=value.shape)
        self._value = value

    def _compute(self):
        pass

    def _compute_backward(self, accumulated_grads):
        pass


class Variable(TensorNode):
    def __init__(self, name='', expected_shape=None):
        TensorNode.__init__(self, name, expected_shape)

    def set_value(self, value):
        self._verify_shape(value)
        self._value = value

    def _compute(self):
        if self._value is None:
            raise RuntimeError('Variable {} needs to be fed'.format(self.name))

    def _compute_backward(self, accumulated_grads):
        pass


def _to_node(val):
    return val if isinstance(val, TensorNode) else Constant(val)


class UnaryOp(TensorNode):
    def __init__(self, argument, name='', expected_shape=None):
        """

        :type argument: TensorNode
        :param name:
        :param expected_shape:
        """
        TensorNode.__init__(self, name, expected_shape)
        if not isinstance(argument, TensorNode):
            raise ValueError('Expected argument to be of type TensorNode')
        self._inputs['argument'] = nodes.InputSlot()
        self._inputs['argument'].set(argument)

    @property
    def argument(self):
        return self._inputs['argument']


class ZerosLike(UnaryOp):

    def __init__(self, argument, name=''):
        argument = _to_node(argument)
        UnaryOp.__init__(self, argument, name=name, expected_shape=argument.expected_shape)

    def _compute(self):
        self._value = np.zeros_like(self.argument.value())

    def _compute_backward(self, accumulated_grads):
        self.argument.set_gradient(np.zeros_like(self._value))


class OnesLike(UnaryOp):

    def __init__(self, argument, name=''):
        argument = _to_node(argument)
        UnaryOp.__init__(self, argument, name=name, expected_shape=argument.expected_shape)

    def _compute(self):
        self._value = np.ones_like(self.argument.value())

    def _compute_backward(self, accumulated_grads):
        self.argument.set_gradient(np.zeros_like(self._value))


class Transpose(UnaryOp):
    def __init__(self, argument, axes=None, name=''):
        argument = _to_node(argument)
        self._axes = axes
        expected_shape = None
        if argument.expected_shape is not None:
            axes = range(len(argument.expected_shape))[::-1] if axes is None else axes
            expected_shape = tuple(argument.expected_shape[i] for i in axes)
        UnaryOp.__init__(self, argument, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = np.transpose(self.argument.value(), axes=self._axes)

    def _compute_backward(self, accumulated_grads):
        original_perm = self._axes
        ndim = len(self.argument.source.shape)
        if self._axes is None:
            original_perm = range(ndim)[::-1]
        grads_perm = [0] * ndim
        for i in range(ndim):
            grads_perm[i] = next(j for j, d in enumerate(original_perm) if d == i)
        self.argument.set_gradient(ops.transpose(accumulated_grads, grads_perm))

    def to_proto(self):
        proto = TensorNode.to_proto(self)
        if self._axes is not None:
            [proto.axes.append(i) for i in self._axes]
        return proto


class Square(UnaryOp):
    def __init__(self, argument, name=''):
        argument = _to_node(argument)
        UnaryOp.__init__(self, argument, name, argument.expected_shape)

    def _compute(self):
        self._value = np.square(self.argument.value())

    def _compute_backward(self, accumulated_grads):
        self.argument.set_gradient(ops.multiply(accumulated_grads, ops.multiply(2., self.argument.source)))


class Sum(UnaryOp):
    def __init__(self, argument, axis=None, keepdims=False, name=''):
        """

        :param argument:
        :param axis: None or int or tuple of ints.
        :param keepdims:
        :param name:
        """
        argument = _to_node(argument)
        expected_shape = None
        if argument.expected_shape is not None:
            if axis is None:
                axis_ = tuple(range(len(argument.expected_shape)))
            else:
                axis_ = axis if not isinstance(axis, int) else (axis,)

            axis_ = sorted((a % len(argument.expected_shape) for a in axis_), reverse=True)
            expected_shape = list(argument.expected_shape)
            if keepdims:
                for a in axis_:
                    expected_shape[a] = 1
            else:
                for a in axis_:
                    expected_shape.pop(a)
        UnaryOp.__init__(self, argument, name, expected_shape)
        self._axis = axis
        self._keepdims = keepdims

    def _compute(self):
        self._value = np.sum(self.argument.value(), axis=self._axis, keepdims=self._keepdims)

    def _compute_backward(self, accumulated_grads):
        # Bunch of ifs to speedup
        output_shape = self.shape.value
        input_shape = self.argument.source.shape.value
        if self._axis is None:
            wires = np.ones(output_shape + input_shape, dtype=float)
        else:
            wires = np.zeros(output_shape + input_shape, dtype=float)

            axis = self._axis if not isinstance(self._axis, int) else (self._axis,)
            axis = sorted((a % len(input_shape) for a in axis), reverse=True)
            for _slice in itertools.product(*(range(n) for n in output_shape)):
                input_slice = list(_slice)
                if self._keepdims:
                    for i in axis:
                        input_slice[i] = slice(None)
                else:
                    for i in axis:
                        input_slice.insert(i, slice(None))
                wires[_slice + tuple(input_slice)] = 1.
        self.argument.set_gradient(ops.tensordot(accumulated_grads, wires, axes=len(self.shape.value)))

    def to_proto(self):
        proto = TensorNode.to_proto(self)
        proto.keepdims = self._keepdims
        if self._axis is not None:
            [proto.axis.append(i) for i in self._axis]
        return proto


class BinaryOp(TensorNode):
    def __init__(self, left, right, name='', expected_shape=None):
        TensorNode.__init__(self, name, expected_shape)
        if not isinstance(left, TensorNode) or not isinstance(right, TensorNode):
            raise ValueError('Expected both arguments are instances of TensorNode')
        for slot_name in ['left', 'right']:
            self._inputs[slot_name] = nodes.InputSlot()
        self._inputs['left'].set(left)
        self._inputs['right'].set(right)

    @property
    def left(self):
        return self._inputs['left']

    @property
    def right(self):
        return self._inputs['right']


def _tie_grads(output_shape, input_shape):
    wire = np.zeros(output_shape + input_shape, dtype=float)

    for input_indices in itertools.product(*(range(n) for n in input_shape)):
        wire_indices = ([None] * len(output_shape)) + list(input_indices)
        dims_diff = len(output_shape) - len(input_shape)
        for idx in range(dims_diff, len(output_shape)):
            out_dim = output_shape[idx]
            in_dim = input_shape[idx - dims_diff]
            if out_dim == in_dim:
                wire_indices[idx] = input_indices[idx - dims_diff]
            else:
                assert in_dim == 1 and input_indices[idx - dims_diff] == 0
                wire_indices[idx] = slice(None)
        for idx in range(dims_diff):
            wire_indices[idx] = slice(None)

        wire[tuple(wire_indices)] = 1.
    return wire


class Add(BinaryOp):
    def __init__(self, left, right, name=''):
        left = _to_node(left)
        right = _to_node(right)
        expected_shape = None
        if left.expected_shape is not None and right.expected_shape is not None:
            expected_shape = np.broadcast_shapes(left.expected_shape, right.expected_shape)
        BinaryOp.__init__(self, left, right, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = self.left.value() + self.right.value()

    def _compute_backward(self, accumulated_grads):
        output_shape = self.value.shape
        for slot in [self.left, self.right]:
            wire = _tie_grads(output_shape, slot.source.value.shape)
            slot.set_gradient(ops.tensordot(accumulated_grads, wire, axes=len(output_shape)))


class Multiply(BinaryOp):
    def __init__(self, left, right, name=''):
        left = _to_node(left)
        right = _to_node(right)
        expected_shape = None
        if left.expected_shape is not None and right.expected_shape is not None:
            expected_shape = np.broadcast_shapes(left.expected_shape, right.expected_shape)
        BinaryOp.__init__(self, left, right, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = np.multiply(self.left.value(), self.right.value())

    def _compute_backward(self, accumulated_grads):
        left_shape = self.left.source.value.shape
        right_shape = self.right.source.value.shape
        output_shape = self.value.shape
        assert (left_shape == output_shape) or (right_shape == output_shape)
        if left_shape == right_shape:
            self.left.set_gradient(ops.multiply(accumulated_grads, self.right.source))
            self.right.set_gradient(ops.multiply(accumulated_grads, self.left.source))
        else:
            small, big = (self.left, self.right) if right_shape == output_shape else (self.right, self.left)
            big.set_gradient(ops.multiply(accumulated_grads, ops.multiply(ops.ones_like(big.source), small.source)))
            small.set_gradient(ops.tensordot(ops.multiply(accumulated_grads, big.source),
                                             _tie_grads(output_shape, small.source.value.shape),
                                             axes=len(output_shape)))


class Matmul(BinaryOp):
    def __init__(self, left, right, name=''):
        left = _to_node(left)
        right = _to_node(right)
        expected_shape = None
        if left.expected_shape is not None and right.expected_shape is not None:
            full_shapes = [len(n.expected_shape) >= 2 for n in (left, right)]
            if all(full_shapes):
                expected_shape = left.expected_shape[:-1] + right.expected_shape[1:]
            elif full_shapes[0]:
                expected_shape = left.expected_shape[:-1]
            elif full_shapes[1]:
                expected_shape = right.expected_shape[1:]
        BinaryOp.__init__(self, left, right, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = np.matmul(self.left.value(), self.right.value())

    def _compute_backward(self, accumulated_grads):
        left_shape = self.left.source.value.shape
        right_shape = self.right.source.value.shape
        if len(left_shape) != 2 or len(right_shape) != 2:
            raise NotImplementedError()

        self.left.set_gradient(ops.matmul(accumulated_grads, ops.transpose(self.right.source)))
        self.right.set_gradient(ops.matmul(ops.transpose(self.left.source), accumulated_grads))


class TensorDot(BinaryOp):
    def __init__(self, left, right, axes=2, name=''):
        """
        Sum product of the last `axes` axes of left with the first `axes` axes of right.
        :param left:
        :param right:
        :param axes:
        :param name:
        """
        # if axes <= 0:
        #     raise ValueError('axes must be positive')
        self._axes = axes
        left = _to_node(left)
        right = _to_node(right)
        expected_shape = None
        if left.expected_shape is not None and right.expected_shape is not None:
            expected_shape = left.expected_shape[:-axes] + right.expected_shape[axes:]
        BinaryOp.__init__(self, left, right, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = np.tensordot(self.left.value(), self.right.value(), axes=self._axes)

    def _compute_backward(self, accumulated_grads):
        # TODO: implement
        raise NotImplementedError('')

    def to_proto(self):
        proto = TensorNode.to_proto(self)
        proto.axes = self._axes
        return proto


class MultiOp(TensorNode):
    def __init__(self, *arguments, name='', expected_shape=None):
        if len(arguments) <= 2:
            raise ValueError('MultiOp is for operations of more than 2 arguments')
        TensorNode.__init__(self, name, expected_shape)
        self._slot_names = []
        for idx, arg in enumerate(arguments):
            if not isinstance(arg, TensorNode):
                raise ValueError('Argument #{} is not a TensorNode'.format(idx))
            slot_name = 'arg_{}'.format(idx)
            self._slot_names.append(slot_name)
            self._inputs[slot_name] = nodes.InputSlot()
            self._inputs[slot_name].set(arg)
            setattr(self, slot_name, property(fget=lambda op: op._inputs[slot_name]))


class SumList(MultiOp):
    def __init__(self, *arguments, name=''):
        arguments = tuple(map(_to_node, arguments))
        expected_shape = None
        if len(arguments) >= 2:
            expected_shape = arguments[0].expected_shape
            for arg in arguments[1:]:
                expected_shape = np.broadcast_shapes(expected_shape,
                                                     () if arg.expected_shape is None else arg.expected_shape)
        MultiOp.__init__(*arguments, name=name, expected_shape=expected_shape)

    def _compute(self):
        self._value = None
        for slot_name in self._slot_names:
            slot_val = getattr(self, slot_name).value()
            self._value = slot_val.copy() if self._value is None else self._value + slot_val
        # shape?

    def _compute_backward(self, accumulated_grads):
        for slot_name in self._slot_names:
            getattr(self, slot_name).set_gradient(ops.multiply(accumulated_grads, 1.))


def _get_inputs(node_data, node_getter):
    result = {}
    for input_slot in node_data.inputs:
        if input_slot.source_name != '':
            result[input_slot.name] = node_getter(input_slot.source_name)
    return result


def from_proto(proto_node, node_getter):
    node_type = proto_node.WhichOneof('node')
    if node_type == Constant.__name__.lower():
        node_data = proto_node.constant.node_data
        value = serializer.from_ndarray(node_data.value)
        return Constant(value, name=node_data.name)
    if node_type == Variable.__name__.lower():
        node_data = proto_node.variable.node_data
        expected_shape = None if len(node_data.expected_shape) == 0 else tuple(node_data.expected_shape)
        v = Variable(name=node_data.name, expected_shape=expected_shape)
        if node_data.HasField('value'):
            v.set_value(serializer.from_ndarray(node_data.value))
        return v
    if node_type == ZerosLike.__name__.lower():
        node_data = proto_node.zeroslike.node_data
        argument = _get_inputs(node_data, node_getter)['argument']
        return ZerosLike(argument=argument, name=node_data.name)
    if node_type == OnesLike.__name__.lower():
        node_data = proto_node.oneslike.node_data
        argument = _get_inputs(node_data, node_getter)['argument']
        return OnesLike(argument=argument, name=node_data.name)
    if node_type == Transpose.__name__.lower():
        node_data = proto_node.transpose.node_data
        argument = _get_inputs(node_data, node_getter)['argument']
        axes = None if len(proto_node.transpose.axes) == 0 else tuple(proto_node.tranpose.axes)
        return Transpose(argument=argument, axes=axes, name=node_data.name)
    if node_type == Square.__name__.lower():
        node_data = proto_node.square.node_data
        argument = _get_inputs(node_data, node_getter)['argument']
        return Square(argument=argument, name=node_data.name)
    if node_type == Sum.__name__.lower():
        node_data = proto_node.sum.node_data
        argument = _get_inputs(node_data, node_getter)['argument']
        axis = None if len(proto_node.sum.axis) == 0 else tuple(proto_node.sum.axis)
        return Sum(argument=argument, axis=axis, keepdims=proto_node.sum.keepdims, name=node_data.name)
    if node_type == Add.__name__.lower():
        node_data = proto_node.add.node_data
        args = _get_inputs(node_data, node_getter)
        return Add(left=args['left'], right=args['right'], name=node_data.name)
    if node_type == Multiply.__name__.lower():
        node_data = proto_node.multiply.node_data
        args = _get_inputs(node_data, node_getter)
        return Multiply(left=args['left'], right=args['right'], name=node_data.name)
    if node_type == Matmul.__name__.lower():
        node_data = proto_node.matmul.node_data
        args = _get_inputs(node_data, node_getter)
        return Matmul(left=args['left'], right=args['right'], name=node_data.name)
    if node_type == TensorDot.__name__.lower():
        node_data = proto_node.tensordot.node_data
        args = _get_inputs(node_data, node_getter)
        return TensorDot(left=args['left'], right=args['right'], axes=proto_node.tensordot.axes, name=node_data.name)
    if node_type == SumList.__name__.lower():
        node_data = proto_node.sumlist.node_data
        args = _get_inputs(node_data, node_getter)
        return SumList(*args, name=node_data.name)
    if node_type is not None:
        raise RuntimeError('Unknown node: {}'.format(str(proto_node)))


def read_graph(proto_graph):
    from sawdown.tensorcube.components import graph

    g = graph.Graph()
    with g:
        for proto_node in proto_graph.nodes:
            _ = from_proto(proto_node, node_getter=g.get_node)

        for proto_node in proto_graph.nodes[::-1]:
            node_type = proto_node.WhichOneof('node')
            node_data = getattr(proto_node, node_type).node_data

            # Gradient of the node
            node = g.get_node(node_data.name)
            if node_data.gradient_name != '':
                node.set_gradient(g.get_node(node_data.gradient_name))

            # Gradient of the input slots
            for input_slot in node_data.inputs:
                if input_slot.grad_name != '':
                    # Good glory God.
                    getattr(node, input_slot.name).set_gradient(g.get_node(input_slot.grad_name))
    return g

from sawdown.tensorcube.components import graph_stack, serializer
from sawdown.proto import tensorcube_pb2


class Named(object):
    def __init__(self, name=''):
        if name == '':
            name = self.__class__.__name__
        g = graph_stack.active_graph()
        if g.has_node(name):
            name = '{}:{}'.format(name, g.size)
        self.name = name
        g.add_node(self)

    def __str__(self):
        return '{}(name={})'.format(self.__class__.__name__, self.name)


class InputSlot(object):
    def __init__(self):
        self._source = None
        self._gradient = None

    @property
    def source(self):
        """

        :rtype: Node
        """
        return self._source

    @property
    def gradient(self):
        return self._gradient

    def set(self, source):
        """

        :type source: Node
        :return:
        """
        if not isinstance(source, Node):
            raise ValueError('Expected a Node object')
        self._source = source

    def set_gradient(self, grad):
        """

        :type grad: tensorcube.components.nodes_tensor.TensorNode
        :return:
        """
        self._gradient = grad

    def value(self):
        if self._source is None:
            raise RuntimeError('Missing input')
        return self._source.value


class Node(Named):

    def __init__(self, name=''):
        Named.__init__(self, name)
        self._inputs = {}
        self._value = None

    @property
    def value(self):
        return self._value

    def parents(self):
        """
        Return an iterable over parent nodes of `self`, each entry has (source_node, dest_slot_name)
        :return:
        """
        return filter(lambda item: item[0] is not None,
                      map(lambda item: (item[1].source, item[0]), self._inputs.items()))

    def evaluate(self):
        for node, _ in graph_stack.active_graph().traverse(self, ()):
            node._evaluate()
        return self._value

    def _evaluate(self):
        raise NotImplementedError()

    def to_proto(self):
        proto = tensorcube_pb2.NodeData(name=self.name)
        if self._value is not None:
            proto.value.CopyFrom(serializer.to_ndarray(self._value))
        for name, slot in self._inputs.items():
            slot_data = tensorcube_pb2.InputSlot(name=name)
            if slot.source is not None:
                slot_data.source_name = slot.source.name
            if slot.gradient is not None:
                slot_data.grad_name = slot.gradient.name
            proto.inputs.append(slot_data)
        return proto


class ShapeNode(Node):
    """
    ShapeNode is a Primitive Node that has a set_value() method,
    i.e. a mutable node, only used internally to provide shape information
    of tensors.
    """
    def __init__(self, name=''):
        Node.__init__(self, name)

    def set_value(self, shape=()):
        self._value = None if shape is None else tuple(shape)

    def _evaluate(self):
        pass

    def _evaluate_backward(self, accumulated_grads):
        pass

    def to_proto(self):
        # Prevent serialization by accident.
        raise NotImplementedError('')

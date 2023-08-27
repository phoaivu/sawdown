import itertools
import numpy as np

from sawdown.tensorcube.components import nodes


class Sequence(nodes.Named):
    """
    Usage: use self.iterator to construct the computation procedure,
    then in the evaluate() function of the aggregation node, keep calling next(self).
    next(self) returns the value of the iterator at each iteration,
     for the sake of compatibility, but it shouldn't be used.
    """
    def __init__(self, name=''):
        nodes.Named.__init__(self, name)

    def iterator(self):
        raise NotImplementedError()

    def __next__(self):
        raise NotImplementedError()

    def _reset(self):
        """
        Reset the iterator, if needed.
        :return:
        """
        pass

    def map(self, func):
        # Execute func, create a FunctionNode (a function of tensors)
        return MappedSequence(self, func)

    def filter(self, predicate):
        return FilteredSequence(self, predicate)

    def packed(self, how):
        # Return a (Tensor)Node depending on `how`.
        pass


class PrimitiveSequence(Sequence):

    def __init__(self, source, dims=None, name=''):
        """

        :type source: tensorcube.components.nodes.TensorNode
        :param dims: either None, an integer, or tuple of integers. Slices are not (yet) supported.
        """
        Sequence.__init__(self, name)
        if dims is not None:
            dims = tuple(dims)
            if any(not isinstance(d, int) for d in dims):
                raise ValueError('Dimensions are to be integers')
            if len(set(dims)) < len(dims):
                raise ValueError('Duplicated entries in dims')
        expected_element_shape = None
        if source.expected_shape is not None and dims is not None:
            source_ndims = len(source.expected_shape)
            if any(d >= source_ndims for d in dims):
                raise ValueError('Invalid dimensions: source expected shape is {}, while dims is {}'.format(
                    source.expected_shape, dims))
            normed_dims = set([d % len(source.expected_shape) for d in dims])
            expected_element_shape = tuple(s for i, s in enumerate(source.expected_shape) if i not in normed_dims)
        self._source = source
        self._dims = dims
        self._iterator = nodes.Variable(self.name + '_iterator', expected_element_shape)
        self._indices = None
        self._slicer = None

    def iterator(self):
        return self._iterator

    def __next__(self):
        if self._indices is None:
            self._source.evaluate()
            if self._dims is None:
                self._indices = itertools.product(*[list(range(s)) for s in self._source.shape.value])
                self._slicer = [0 for _ in self._source.shape]
            else:
                self._indices = itertools.product(*[list(range(self._source.shape.value[d])) for d in self._dims])
                self._slicer = [slice(0, None) for _ in self._source.shape]

        try:
            pos = next(self._indices)
        except StopIteration as e:
            self._reset()
            raise e

        if self._dims is None:
            self._slicer = pos
        else:
            for d, v in zip(self._dims, pos):
                self._slicer[d] = v

        iterator_val = self._source.value[self._slicer]
        self._iterator.set_value(iterator_val)
        return iterator_val

    def _reset(self):
        self._indices = None


class ZippedSequence(Sequence):

    def __init__(self, first, second, *more, name=''):
        """

        :type first: Sequence
        :type second: Sequence
        :type more: tuple[Sequence]
        :param name:
        """
        Sequence.__init__(self, name)
        self._sources = [first, second] + list(more)
        self._iterators = tuple(self._flatten([s.iterator() for s in self._sources]))

    def _flatten(self, items):
        result = []
        for i in items:
            if isinstance(i, tuple):
                [result.append(ii) for ii in i]
            else:
                result.append(i)
        return result

    def iterator(self):
        return self._iterators

    def __next__(self):
        try:
            values = []
            for source in self._sources:
                values.append(next(source))
            return self._flatten(values)
        except StopIteration as e:
            self._reset()
            raise e

    def _reset(self):
        [s._reset() for s in self._sources]


class FunctionatedSequence(Sequence):
    """
    A sequence created from another Sequence and a function, could be a filter or a map.
    """
    def __init__(self, source, func, name=''):
        Sequence.__init__(self, name)
        self._source = source
        source_iterator = self._source.iterator()
        if isinstance(source_iterator, tuple):
            self._output = func(*source_iterator)
        else:
            self._output = func(source_iterator)
        if isinstance(self._output, (tuple, list)):
            if any(not isinstance(o, nodes.Node) for o in self._output):
                raise ValueError('Ease your life by not nesting things in the output of func')
            self._output = tuple(self._output)
        else:
            assert isinstance(self._output, nodes.Node)

    def _reset(self):
        self._source._reset()


class MappedSequence(FunctionatedSequence):
    def __init__(self, source, func, name=''):
        """

        :type source: Sequence
        :param func:
        """
        FunctionatedSequence.__init__(self, source, func, name)

    def iterator(self):
        return self._output

    def __next__(self):
        try:
            next(self._source)
        except StopIteration as e:
            self._reset()
            raise e
        if isinstance(self._output, tuple):
            values = []
            for o in self._output:
                o.evaluate()
                values.append(o.value)
            values = tuple(values)
        else:
            self._output.evaluate()
            values = self._output.value
        return values


class FilteredSequence(FunctionatedSequence):

    def __init__(self, source, predicate, name=''):
        FunctionatedSequence.__init__(self, source, predicate, name)
        if not isinstance(self._output, nodes.Node):
            raise ValueError('predicated should return a single TensorNode of type boolean')

    def iterator(self):
        return self._source.iterator()

    def __next__(self):
        satisfied = False
        while not satisfied:
            try:
                next(self._source)
            except StopIteration as e:
                self._reset()
                raise e
            self._output.evaluate()
            if self._output.shape.value != () and self._output.shape.value != (1,):
                raise ValueError('predicate should return a single boolean')
            satisfied = np.all(self._output.value)

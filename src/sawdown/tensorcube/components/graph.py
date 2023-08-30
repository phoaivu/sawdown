import collections
import queue

from sawdown.tensorcube.components import graph_stack
from sawdown.proto import tensorcube_pb2


class Graph(object):

    def __init__(self):
        # nodes[node_name] -> node
        self._nodes = dict()
        self._paths = {}

    @property
    def size(self):
        return len(self._nodes)

    def has_node(self, node_or_name):
        if isinstance(node_or_name, str):
            return node_or_name in self._nodes.keys()
        return node_or_name in self._nodes.values()

    def add_node(self, node):
        assert node.name not in self._nodes.keys(), 'Duplicated node name: {}'.format(node.name)
        self._nodes[node.name] = node
        self._paths.clear()
        return node

    def get_node(self, name):
        return self._nodes[name]

    def traverse(self, start_node, end_nodes):
        """
        Topologically sort the ancestors of start_node, up to all the nodes in `end_nodes`.
        Raise RuntimeError if cannot reach `end_nodes`.
        :type start_node: tensorcube.components.nodes.Node
        :type end_nodes: tuple[tensorcube.components.nodes.Node]
        :return: a list of tuple (source_node, [dest1_slot, dest2_slot, ...])
        """
        path_key = (start_node.name, ) + tuple(n.name for n in end_nodes)
        if path_key not in self._paths:
            self._paths[path_key] = self._traverse(start_node, end_nodes)
        return self._paths[path_key]

    def _traverse(self, start_node, end_nodes):
        """
        Topologically sort the ancestors of start_node, up to all the nodes in `end_nodes`.
        Raise RuntimeError if cannot reach `end_nodes`.
        :type start_node: tensorcube.components.nodes.Node
        :type end_nodes: tuple[tensorcube.components.nodes.Node]
        :return: a list of tuple (source_node, [dest1_slot, dest2_slot, ...])
        """
        assert start_node in self._nodes.values()
        assert all(n in self._nodes.values() for n in end_nodes)

        # Get all the nodes and edges in the evaluation path
        nodes = set()
        edges = collections.defaultdict(set)
        for node in end_nodes:
            for p, slot_name in node.parents():
                edges[p.name].add(getattr(node, slot_name))
        remaining = queue.SimpleQueue()
        remaining.put(start_node)
        while not remaining.empty():
            node = remaining.get()
            nodes.add(node)

            remaining_parents = list(filter(lambda item: (node.name, item[1]) not in edges.get(item[0].name, set()),
                                            node.parents()))
            if len(remaining_parents) > 0:
                for p, slot_name in remaining_parents:
                    edges[p.name].add(getattr(node, slot_name))
                    remaining.put(p)

        silo_nodes = set(e for e in end_nodes if e not in nodes)
        if len(silo_nodes) > 0:
            raise RuntimeError('Cannot reach {} from {}'.format(start_node, silo_nodes))

        sorted_nodes = []
        consumed_edges = collections.defaultdict(set)
        remaining = queue.SimpleQueue()
        remaining.put(start_node)

        while not remaining.empty():
            node = remaining.get()
            assert len(edges.get(node.name, set())) == 0
            sorted_nodes.append((node, list(consumed_edges[node.name])))
            nodes.remove(node)

            parents = list(node.parents())
            for p, slot_name in parents:
                dest_slot = getattr(node, slot_name)
                edges[p.name].remove(dest_slot)
                consumed_edges[p.name].add(dest_slot)
            for p, slot_name in parents:
                if len(edges.get(p.name, set())) == 0:
                    remaining.put(p)
        if len(nodes) > 0:
            raise RuntimeError('Cycle detected around {}'.format(nodes))
        return sorted_nodes[::-1]

    def __enter__(self):
        graph_stack.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        graph_stack.pop()

    def to_proto(self):
        from sawdown.tensorcube.components import nodes_tensor
        proto = tensorcube_pb2.Graph()
        if self.size == 0:
            return proto

        # properly sort things
        nodes = []
        unsorted_nodes = []
        edges = collections.defaultdict(set)
        remaining = queue.SimpleQueue()
        for node in self._nodes.values():
            parents = list(node.parents())
            if len(parents) == 0:
                remaining.put(node)
            else:
                unsorted_nodes.append(node)
                for p, slot_name in parents:
                    edges[p.name].add((node.name, slot_name))

        while not remaining.empty():
            node = remaining.get()
            nodes.append(node)
            if node.name in edges:
                edges.pop(node.name)

            abandoned_nodes = []
            for i, node in enumerate(unsorted_nodes):
                clean = not any((node.name, slot_name) in edges.get(p.name, set()) for p, slot_name in node.parents())
                if clean:
                    abandoned_nodes.append(i)
            for i in abandoned_nodes[::-1]:
                remaining.put(unsorted_nodes[i])
                unsorted_nodes.pop(i)

        assert len(unsorted_nodes) == 0, 'There are loops in the graph around {}'.format(unsorted_nodes)

        for node in filter(lambda n: isinstance(n, nodes_tensor.TensorNode), nodes):
            node_dict = dict()
            node_dict[node.__class__.__name__.lower()] = node.to_proto()
            proto.nodes.append(tensorcube_pb2.Node(**node_dict))
        return proto

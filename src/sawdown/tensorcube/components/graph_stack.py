import threading


class GraphStack(threading.local):
    def __init__(self):
        self._graphs = []

    def push(self, g):
        self._graphs.insert(0, g)

    def pop(self):
        self._graphs.pop(0)

    def head(self):
        return self._graphs[0]


_graph_stack = GraphStack()


def push(g):
    _graph_stack.push(g)


def pop():
    _graph_stack.pop()


def active_graph():
    return _graph_stack.head()

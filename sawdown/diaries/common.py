import collections
import enum
import importlib

import numpy as np
from sawdown.diaries import readers


class Termination(enum.IntEnum):
    CONTINUE = 0
    FAILED_INITIALIZATION = 1
    MAX_ITERATION = 2
    INFINITESIMAL_STEP = 3
    # Used in LinearInitialization, when x_k satisfies the linear constraints.
    SATISFIED = 4
    # Used in MetaOptimizers, indicating failed solving subproblem, usually due to exceptions
    FAILED = 5


class RecordTypes(enum.IntEnum):
    PRE_RUN = 1
    ITERATION = 2
    SOLUTION = 3
    POST_RUN = 4


class Solution(collections.OrderedDict):
    def __init__(self, x=None, iteration=-1, objective=np.nan, termination=Termination.FAILED, **kwargs):
        super().__init__(x=x, iteration=iteration, objective=objective,
                         termination=termination, termination_name=str(termination), **kwargs)
        self._iteration_data = None

    @property
    def x(self):
        return self['x']

    @property
    def iteration(self):
        return self['iteration']

    @property
    def objective(self):
        return self['objective']

    @property
    def termination(self):
        return self['termination']

    @property
    def termination_name(self):
        return self['termination_name']

    def set_iteration_data(self, iteration_data):
        self._iteration_data = iteration_data

    def set_reader_config(self, **reader_config):
        for k, v in reader_config.items():
            self['__{}__'.format(k)] = v

    def iteration_data_reader(self):
        if self._iteration_data is not None:
            return readers.MemoryReader(records=self._iteration_data)

        if all(k in self for k in ['__reader_module__', '__reader_class__', '__reader_args__']):
            module = importlib.import_module(self['__reader_module__'])
            return getattr(module, self['__reader_class__'])(*self['__reader_args__'])
        raise RuntimeError('Reader configuration was not written, likely not supported by the writer.')
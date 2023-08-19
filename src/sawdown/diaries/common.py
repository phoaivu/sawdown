import collections
import enum
import importlib

import numpy as np
from sawdown.diaries import readers


class DiaryWorkerMessageType(enum.IntEnum):
    ENTRY = 1
    STOP = 2
    REQUEST_ITERATION_DATA = 3
    REQUEST_READER_CONFIG = 4
    CLOSE_DIARY = 5


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

        pickle_file = self.get('__pickle_file__', None)
        if pickle_file is not None:
            return readers.MemoryReader(pickle_file=pickle_file)

        raise RuntimeError('Reader configuration was not written, likely not supported by the writer.')

import collections
import datetime
import threading

from sawdown.diaries import common


class RollingLog(object):

    def __init__(self):
        self._record_type = common.RecordTypes.PRE_RUN
        self._iteration = -1
        self._active_record = collections.OrderedDict()

    def _reset(self):
        self._record_type = common.RecordTypes.PRE_RUN
        self._iteration = -1
        self._active_record = collections.OrderedDict()

    def as_long_as(self, condition):
        self.flush()
        self._iteration = 0
        self._record_type = common.RecordTypes.ITERATION
        while condition():
            yield self._iteration
            self.flush()
            self._iteration += 1
        self._record_type = common.RecordTypes.POST_RUN

    def __setitem__(self, key, value):
        self._active_record[key] = value

    def set_items(self, **kwargs):
        self._active_record.update(**kwargs)

    def append_items(self, **kwargs):
        """
        Like set_items() but make the entries into a list.
        :param kwargs:
        :return:
        """
        for k, v in kwargs.items():
            existing_val = self._active_record.get(k, [])
            if isinstance(existing_val, list):
                existing_val.append(v)
            else:
                existing_val = [existing_val, v]
            self._active_record[k] = existing_val

    def flush(self):
        if len(self._active_record) > 0:
            ordered_record = collections.OrderedDict(iteration=self._iteration, record_type=self._record_type,
                                                     record_type_name=str(self._record_type))
            ordered_record.update(**self._active_record)
            self.write_record(ordered_record)
            self._active_record = collections.OrderedDict()

    def write_record(self, record=None):
        raise NotImplementedError()


class DiaryStack(threading.local):
    def __init__(self):
        super().__init__()
        self.stack = []

    def push(self, diary):
        self.stack.append(diary)

    def pop(self):
        return self.stack.pop(len(self.stack) - 1)

    def head(self):
        return None if len(self.stack) == 0 else self.stack[-1]


class DiaryBase(RollingLog):

    _diary_stack = DiaryStack()

    def __init__(self, diary_id=''):
        """
        """
        RollingLog.__init__(self)
        if diary_id in {'', None}:
            diary_id = '0'
        self._diary_id = diary_id
        self._parent_id = '_'.join(self._diary_id.split('_')[:-1])
        self._sub_count = 0
        self.solution = None
        self._starting_time = None
        self._records = []
        self._descendant_iteration_data = collections.OrderedDict()

    @property
    def identifier(self):
        return self._diary_id

    def is_root(self):
        return self._parent_id in {'', None}

    def iteration_data(self):
        """
        Iteration data of this diary alone.
        :return:
        """
        return {self._diary_id: (self._parent_id, self._records)}

    def descendant_iteration_data(self):
        """
        Iteration data of all diaries descended from this, including its own data.
        :return:
        """
        all_iteration_data = collections.OrderedDict()
        all_iteration_data.update(self.iteration_data())
        all_iteration_data.update(self._descendant_iteration_data)
        return all_iteration_data

    def __enter__(self):
        self._reset()
        self.solution = None
        self._starting_time = datetime.datetime.now()
        self.set_items(starting_time=self._starting_time)
        self._records = []
        self._iteration_data = collections.OrderedDict()
        self._initialize()
        DiaryBase._diary_stack.push(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        self._starting_time = None
        if self.solution is not None:
            self.solution.set_iteration_data(self.descendant_iteration_data())
        DiaryBase._diary_stack.pop()

        # Give iteration data to the parent
        parent = DiaryBase._diary_stack.head()
        if parent is not None:
            parent._descendant_iteration_data.update(**self.descendant_iteration_data())

    def set_solution(self, x, objective, termination=common.Termination.MAX_ITERATION, **kwargs):
        self.flush()
        self.solution = common.Solution(x=x, iteration=self._iteration, objective=objective, termination=termination,
                                        record_type=common.RecordTypes.SOLUTION,
                                        record_type_name=str(common.RecordTypes.SOLUTION),
                                        spent_time=datetime.datetime.now() - self._starting_time,
                                        **kwargs)
        self.solution.set_reader_config(**self._get_reader_config())
        self.write_record(self.solution)

    def sub(self):
        self._sub_count += 1
        sub = self._dup('{}_{}'.format(self._diary_id, self._sub_count - 1))
        self.append_items(sub_diary_index=sub.identifier)
        return sub

    def write_record(self, record=None):
        if self._starting_time is None:
            raise ValueError('Use a `with` context to initialize diary writer.')
        self._records.append(record)
        self._write(record)

    def _dup(self, new_diary_id=''):
        raise NotImplementedError()

    def _initialize(self):
        pass

    def _close(self):
        pass

    def _write(self, entries=None):
        raise NotImplementedError()

    def _get_reader_config(self):
        raise NotImplementedError()

import collections

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


class DiaryBase(RollingLog):

    def __init__(self, diary_id=''):
        RollingLog.__init__(self)
        if diary_id in {'', None}:
            diary_id = '0'
        self._diary_id = diary_id
        self._parent_id = '_'.join(self._diary_id.split('_')[:-1])
        self._sub_count = 0

    def is_root(self):
        return self._parent_id == ''

    @property
    def identifier(self):
        return self._diary_id

    def set_solution(self, x, objective, termination=common.Termination.MAX_ITERATION, **kwargs):
        raise NotImplementedError()

    def sub(self):
        raise NotImplementedError()

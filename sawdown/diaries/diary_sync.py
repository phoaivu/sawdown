import datetime

from sawdown.diaries import base, writers, common


class SyncDiary(base.DiaryBase):

    def __init__(self, diary_id, proto_diaries, parent_writers=None):
        base.DiaryBase.__init__(diary_id)
        if parent_writers is None:
            parent_writers = writers.from_proto(proto_diaries)
        self._writers = parent_writers
        self._starting_time = None
        self.solution = None

    def __enter__(self):
        self._reset()
        self._starting_time = datetime.datetime.now()
        self.solution = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._starting_time = None
        if self.is_root() and self.solution is not None:
            for writer in self._writers:
                if not hasattr(writer, 'get_iteration_data'):
                    continue
                iteration_data = writer.get_iteration_data(self._diary_id)
                if iteration_data is not None:
                    self.solution.set_iteration_data(iteration_data)
                    break

    def set_solution(self, x, objective, termination=common.Termination.MAX_ITERATION, **kwargs):
        self.flush()
        self.solution = common.Solution(x=x, iteration=self._iteration, objective=objective, termination=termination,
                                        record_type=common.RecordTypes.SOLUTION,
                                        record_type_name=str(common.RecordTypes.SOLUTION),
                                        spent_time=datetime.datetime.now() - self._starting_time,
                                        **kwargs)
        for writer in self._writers:
            if not hasattr(writer, 'get_reader_config'):
                continue
            reader_config = writer.get_reader_config(self._diary_id)
            if len(reader_config) > 0:
                self.solution.set_reader_config(**reader_config)
                break
        self.write_record(self.solution)

    def sub(self):
        self._sub_count += 1
        return SyncDiary('{}_{}'.format(self._diary_id, self._sub_count - 1), proto_diaries=None,
                         parent_writers=self._writers)

    def write_record(self, record=None):
        if self._starting_time is None:
            raise ValueError('Use a `with` context to initialize diary writer.')
        for writer in self._writers:
            writer.write(self._diary_id, self._parent_id, record)

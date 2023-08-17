import datetime
import multiprocessing

from sawdown.diaries import base, common, writers


class DiaryWorker(multiprocessing.Process):

    def __init__(self, message_queue, response_queue, writer_config=()):
        multiprocessing.Process.__init__(self, name=self.__class__.__name__)
        self._message_queue = message_queue
        self._response_queue = response_queue
        self._writer_config = writer_config

    def run(self):
        _writers = writers.from_proto(self._writer_config)

        (msg_type, msg) = self._message_queue.get()
        while msg_type != common.DiaryWorkerMessageType.STOP:
            if msg_type == common.DiaryWorkerMessageType.ENTRY:
                (diary_id, parent_id, entries) = msg
                [w.write(diary_id, parent_id, entries) for w in _writers]
            elif msg_type == common.DiaryWorkerMessageType.REQUEST_ITERATION_DATA:
                writer = next((w for w in _writers if hasattr(w, 'get_iteration_data')), None)
                if writer is None:
                    self._response_queue.put((msg, None))
                else:
                    self._response_queue.put((msg, writer.get_iteration_data(diary_id=msg)))
            elif msg_type == common.DiaryWorkerMessageType.REQUEST_READER_CONFIG:
                writer = next((w for w in _writers if hasattr(w, 'get_reader_config')), None)
                if writer is None:
                    self._response_queue.put((msg, dict()))
                else:
                    self._response_queue.put((msg, writer.get_reader_config(diary_id=msg)))
            (msg_type, msg) = self._message_queue.get()

        [w.close() for w in _writers]

    def close(self):
        self._message_queue.put((common.DiaryWorkerMessageType.STOP, None))
        self.join()
        multiprocessing.Process.close(self)


class AsyncDiary(base.DiaryBase):

    def __init__(self, diary_id, message_queue, response_queue, response_semaphore):
        base.DiaryBase.__init__(self, diary_id)
        self._message_queue = message_queue
        self._response_queue = response_queue
        self._response_semaphore = response_semaphore
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
            self._response_semaphore.acquire()
            try:
                self._message_queue.put((common.DiaryWorkerMessageType.REQUEST_ITERATION_DATA, self._diary_id))
                self.solution.set_iteration_data(self._response_queue.get()[1])
            finally:
                self._response_semaphore.release()
        self._message_queue = None
        self._response_queue = None

    def set_solution(self, x, objective, termination=common.Termination.MAX_ITERATION, **kwargs):
        self.flush()
        self.solution = common.Solution(x=x, iteration=self._iteration, objective=objective, termination=termination,
                                        record_type=common.RecordTypes.SOLUTION,
                                        record_type_name=str(common.RecordTypes.SOLUTION),
                                        spent_time=datetime.datetime.now() - self._starting_time,
                                        **kwargs)
        self._response_semaphore.acquire()
        try:
            self._message_queue.put((common.DiaryWorkerMessageType.REQUEST_READER_CONFIG, self._diary_id))
            self.solution.set_reader_config(**self._response_queue.get()[1])
        finally:
            self._response_semaphore.release()
        self.write_record(self.solution)

    def sub(self):
        return AsyncDiary(self.new_sub_id(), self._message_queue, self._response_queue, self._response_semaphore)

    def new_sub_id(self):
        self._sub_count += 1
        new_sub_id = '{}_{}'.format(self._diary_id, self._sub_count - 1)
        self.append_items(sub_diary_index=new_sub_id)
        return new_sub_id

    def write_record(self, record=None):
        if self._starting_time is None:
            raise ValueError('Use a `with` context to initialize diary writer.')
        self._message_queue.put((common.DiaryWorkerMessageType.ENTRY, (self._diary_id, self._parent_id, record)))

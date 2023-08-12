import collections
import enum
import os.path
import datetime
import multiprocessing
import pickle
import sys

from sawdown.diaries import base, common, readers


class MessageType(enum.IntEnum):
    ENTRY = 1
    STOP = 2
    REQUEST_ITERATION_DATA = 3
    REQUEST_READER_CONFIG = 4


class AsyncDiary(base.RollingLog):

    def __init__(self, diary_id=''):
        base.RollingLog.__init__(self)
        if diary_id in {'', None}:
            diary_id = '0'
        self._diary_id = diary_id
        self._parent_id = '_'.join(self._diary_id.split('_')[:-1])
        self._sub_count = 0
        self._starting_time = None
        self._message_queue = None
        self._response_queue = None
        self.solution = None

    def is_root(self):
        return self._parent_id == ''

    @property
    def identifier(self):
        return self._diary_id

    def set_queues(self, message_queue, response_queue):
        self._message_queue = message_queue
        self._response_queue = response_queue

    def __enter__(self):
        self._reset()
        self.solution = None
        self._starting_time = datetime.datetime.now()
        if self._message_queue is None:
            raise ValueError('message_queue is not set')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._starting_time = None
        if self.is_root() and self.solution is not None:
            self._message_queue.put((MessageType.REQUEST_ITERATION_DATA, None))
            self.solution.set_iteration_data(self._response_queue.get())
        self._message_queue = None
        self._response_queue = None

    def set_solution(self, x, objective, termination=common.Termination.MAX_ITERATION, **kwargs):
        self.flush()
        self.solution = common.Solution(x=x, iteration=self._iteration, objective=objective, termination=termination,
                                        record_type=common.RecordTypes.SOLUTION,
                                        record_type_name=str(common.RecordTypes.SOLUTION),
                                        spent_time=datetime.datetime.now() - self._starting_time,
                                        **kwargs)
        self._message_queue.put((MessageType.REQUEST_READER_CONFIG, None))
        self.solution.set_reader_config(**self._response_queue.get())
        self.write_record(self.solution)

    def sub(self, with_queues=True):
        self._sub_count += 1
        sub = AsyncDiary('{}_{}'.format(self._diary_id, self._sub_count - 1))
        self.append_items(sub_diary_index=sub.identifier)
        if with_queues:
            sub.set_queues(self._message_queue, self._response_queue)
        return sub

    def write_record(self, record=None):
        if self._starting_time is None:
            raise ValueError('Use a `with` context to initialize diary writer.')
        self._message_queue.put((MessageType.ENTRY, (self._diary_id, self._parent_id, record)))


class AsyncWorkerBase(multiprocessing.Process):

    def __init__(self, message_queue=None, response_queue=None):
        multiprocessing.Process.__init__(self, name=self.__class__.__name__)
        self._message_queue = message_queue
        self._response_queue = response_queue

    def run(self):
        iteration_data = collections.OrderedDict()
        self._initialize()
        (msg_type, msg) = self._message_queue.get()
        while msg_type != MessageType.STOP:
            if msg_type == MessageType.ENTRY:
                (diary_id, parent_id, entries) = msg
                if diary_id not in iteration_data:
                    iteration_data[diary_id] = (parent_id, [])
                iteration_data[diary_id][1].append(entries)
                self._write(diary_id, parent_id, entries)
            elif msg_type == MessageType.REQUEST_ITERATION_DATA:
                self._response_queue.put(iteration_data)
            else:
                assert msg_type == MessageType.REQUEST_READER_CONFIG
                self._response_queue.put(self._get_reader_config())
            (msg_type, msg) = self._message_queue.get()
        self._close(iteration_data)

    def close(self):
        self._message_queue.put((MessageType.STOP, None))
        self.join()
        multiprocessing.Process.close(self)

    def _initialize(self):
        pass

    def _close(self, iteration_data):
        pass

    def _write(self, diary_id='', parent_id='', entries=None):
        raise NotImplementedError()

    def _get_reader_config(self):
        return {}


class StreamDiaryWorker(AsyncWorkerBase):
    def __init__(self, message_queue, response_queue, stream=''):
        AsyncWorkerBase.__init__(self, message_queue, response_queue)
        self._stream = '' if stream is None else stream
        if self._stream not in {'', 'stdout', 'stderr'}:
            raise ValueError('Invalid stream: {}'.format(stream))
        self._output_stream = None
        self._encountered = set()

    def _initialize(self):
        if self._stream != '':
            self._output_stream = getattr(sys, self._stream)
        self._encountered = set()

    def _close(self, iteration_data):
        pass

    def _write(self, diary_id='', parent_id='', entries=None):
        if self._output_stream is None:
            return
        if diary_id not in self._encountered:
            self._encountered.add(diary_id)
            self._output_stream.write('Diary #{}, parent = {}\n'.format(diary_id, parent_id))
        for k, v in entries.items():
            self._output_stream.write('{}: {}\n'.format(k, v))
        self._output_stream.write('------------------\n')


class FileDiaryWorker(AsyncWorkerBase):
    def __init__(self, message_queue, response_queue, path='.', job_name='optimization'):
        AsyncWorkerBase.__init__(self, message_queue, response_queue)
        self._path = path
        self._job_name = job_name
        self._files = {}
        self._log_path = ''

    def _initialize(self):
        self._path = os.path.abspath(self._path)
        i = 0
        refined_job_name = self._job_name
        while os.path.exists(os.path.join(self._path, refined_job_name)):
            refined_job_name = '{}_{}'.format(self._job_name, i)
            i += 1
        self._job_name = refined_job_name
        self._log_path = os.path.join(self._path, self._job_name)
        os.mkdir(self._log_path)

    def _close(self, iteration_data):
        with open(os.path.join(self._log_path, '{}.pkl'.format(self._job_name)), 'wb') as f:
            pickle.dump(iteration_data, f)

    def _write(self, diary_id='', parent_id='', entries=None):
        log_file = self._files.get(diary_id, None)
        if log_file is None:
            log_file = open(os.path.join(self._log_path, '{}.log'.format(diary_id)), 'w')
            log_file.write('Diary #{}, parent = {}\n'.format(diary_id, parent_id))
            self._files[diary_id] = log_file
        for k, v in entries.items():
            log_file.write('{}: {}\n'.format(k, v))
        log_file.write('------------------\n')

    def _get_reader_config(self):
        return dict(reader_module=readers.MemoryReader.__module__,
                    reader_class=readers.MemoryReader.__name__,
                    reader_args=(None, os.path.join(self._log_path, '{}.pkl'.format(self._job_name))))

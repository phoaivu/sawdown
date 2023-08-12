import multiprocessing

from sawdown.diaries import writers, async_diary


def log_stream(stream='stdout'):
    return writers.StreamWriter('', stream)


def log_file(path='.', job_name='optimization'):
    return writers.FileWriter('', path, job_name)


class DiaryMixIn(object):

    def __init__(self):
        self._diary = None

    def _setup(self, objective, opti_math, **kwargs):
        pass

    def log_stream(self, stream='stdout'):
        return self.log(log_stream(stream))

    def log_file(self, path='.', job_name='optimization'):
        return self.log(log_file(path, job_name))

    def log(self, _diary):
        self._diary = _diary
        return self


class AsyncDiaryMixIn(object):
    def __init__(self):
        self._diary = None
        self._diary_worker = None
        self._diary_worker_config = (None, ())
        self._diary_message, self._diary_response = (None, None)

    def _setup(self, objective, opti_math, **kwargs):
        pass

    def _start_diary_worker(self):
        self._diary_message, self._diary_response = (multiprocessing.Queue(), multiprocessing.Queue())
        if self._diary_worker_config[0] is None:
            self._diary_worker_config = (async_diary.StreamDiaryWorker, ('', ))
        self._diary_worker = self._diary_worker_config[0](self._diary_message, self._diary_response,
                                                          *self._diary_worker_config[1])
        self._diary_worker.start()
        self._diary = async_diary.AsyncDiary(diary_id='')
        self._diary.set_queues(self._diary_message, self._diary_response)

    def _stop_diary_worker(self):
        self._diary_worker.close()
        self._diary_worker = None
        self._diary = None
        self._diary_message, self._diary_response = (None, None)

    def log_file(self, path='.', job_name='optimization'):
        self._diary_worker_config = (async_diary.FileDiaryWorker, (path, job_name))
        return self

    def log_stream(self, stream=''):
        self._diary_worker_config = (async_diary.StreamDiaryWorker, (stream, ))
        return self

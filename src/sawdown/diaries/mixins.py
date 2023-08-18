import multiprocessing

from sawdown.proto import sawdown_pb2
from sawdown.diaries import diary_async, diary_sync


class AsyncDiaryMixIn(object):
    def __init__(self, proto_problem):
        self._diary = None
        self._diary_worker = None
        self._diary_message, self._diary_response = (None, None)
        self._diary_response_semaphore = None
        self._diary_writer_config = []
        for msg in proto_problem.diaries:
            if msg.WhichOneof('diary') is not None:
                diary_conf = sawdown_pb2.Diary()
                diary_conf.CopyFrom(msg)
                self._diary_writer_config.append(diary_conf)

    def _start_diary_worker(self):
        self._diary_message, self._diary_response = (multiprocessing.Queue(), multiprocessing.Queue())
        self._diary_response_semaphore = multiprocessing.RLock()
        self._diary_worker = diary_async.DiaryWorker(self._diary_message, self._diary_response,
                                                     self._diary_writer_config)
        self._diary_worker.start()
        self._diary = diary_async.AsyncDiary('', self._diary_message, self._diary_response,
                                             self._diary_response_semaphore)

    def _stop_diary_worker(self):
        self._diary_worker.close()
        self._diary_worker = None
        self._diary = None
        self._diary_message, self._diary_response = (None, None)
        self._diary_response_semaphore = None


def test_diary(stream='stdout'):
    """
    Only useful for tests.
    """
    return diary_sync.SyncDiary(diary_id='',
                                proto_diaries=[sawdown_pb2.Diary(stream_diary=sawdown_pb2.StreamDiary(stream=stream))])

import multiprocessing
import mmap
import pickle


class PickleQueue(object):
    def __init__(self, size_in_bytes=10485760):
        # positions are in bytes.
        self._read_pos = multiprocessing.RawValue('Q', 0)
        self._write_pos = multiprocessing.RawValue('Q', 0)
        self._continue = multiprocessing.RawValue('b', 1)
        self._file = mmap.mmap(-1, size_in_bytes)
        self._file_semaphore = multiprocessing.Semaphore(0)
        self._pos_semaphore = multiprocessing.Semaphore(0)

    def _write_u_long(self, n):
        assert n >= 0
        for i in range(8):
            self._file.write_byte((n >> (8*i)) & 0xFF)

    def _read_u_long(self):
        n = 0
        for i in range(8):
            n |= (self._file.read_byte() << (8*i))
        return n

    def _refresh(self):
        # Delete the read data from the file, and move them up.
        self._file_semaphore.acquire()
        self._file.seek(self._read_pos.value)
        data = self._file.read(self._write_pos.value - self._read_pos.value)
        self._file.seek(0)
        self._file.write(data)
        self._read_pos.value = 0
        self._write_pos.value = len(data)
        self._file_semaphore.release()

    def _writable_wait(self, n_bytes=0):
        while self._continue.value > 0:
            self._pos_semaphore.acquire()
            try:
                if self._file.size() - self._write_pos.value >= n_bytes:
                    return
                if self._read_pos.value > n_bytes:
                    self._refresh()
                else:
                    continue
            finally:
                self._pos_semaphore.release()

    def _readable_wait(self):
        while self._continue.value > 0:
            self._pos_semaphore.acquire()
            try:
                if self._write_pos.value - self._read_pos.value > 8:
                    return
                else:
                    continue
            finally:
                self._pos_semaphore.release()

    def put(self, data):
        binary_data = pickle.dumps(data)
        data_len = len(binary_data)
        if 8 + data_len > self._file.size():
            raise ValueError('Too big to handle. Try increasing queue size in bytes')
        while self._continue.value > 0:
            self._writable_wait(8 + data_len)
            if self._file_semaphore.acquire(False):
                self._file.seek(self._write_pos.value)
                self._write_u_long(data_len)
                self._file.write(binary_data)
                self._write_pos.value = self._file.tell()
                self._file_semaphore.release()
                break

    def get(self):
        data = None
        while self._continue.value > 0:
            self._readable_wait()
            if self._file_semaphore.acquire(False):
                self._file.seek(self._read_pos.value)
                data = self._file.read(self._read_u_long())
                self._read_pos.value = self._file.tell()
                self._file_semaphore.release()
                break
        return None if data is None else pickle.loads(data)

    def close(self):
        self._file_semaphore.acquire()
        self._continue.value = 0
        self._file.close()
        self._file_semaphore.release()


class Silo(multiprocessing.Process):
    """
    A silo is a process communicating via socket.
    """
    pass
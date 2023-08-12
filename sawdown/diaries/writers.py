import os.path
import pickle
import sys

from sawdown.diaries import base, readers


class StreamWriter(base.DiaryBase):
    """
    Store data and optionally write it to a stream.
    """

    def __init__(self, diary_id='', stream=''):
        base.DiaryBase.__init__(self, diary_id)
        self._stream = stream
        if stream not in {'stdout', 'stderr', ''}:
            raise ValueError('`stream` either empty, `stdout` or `stderr`')
        self._output = getattr(sys, stream, None)
        self._first_entry = True

    def _dup(self, new_diary_id=''):
        return StreamWriter(new_diary_id, self._stream)

    def _write(self, entries=None):
        if self._output is None:
            return
        if self._first_entry:
            self._output.write('Diary #{}, Parent={}\n'.format(self._diary_id, self._parent_id))
            self._first_entry = False
        prefix = '' if self._parent_id is None else '    '
        for k, v in entries.items():
            self._output.write('{}{}: {}\n'.format(prefix, k, v))
        self._output.write('-------------------\n')

    def _get_reader_config(self):
        return {}


class FileWriter(base.DiaryBase):
    def __init__(self, diary_id='', path='.', job_name='optimization'):
        base.DiaryBase.__init__(self, diary_id)
        self._path = path
        self._job_name = job_name
        self._output_dir = ''

    def _dup(self, new_diary_id=''):
        return FileWriter(new_diary_id, self._path, self._job_name)

    def _initialize(self):
        self._path = os.path.abspath(self._path)
        if not os.path.exists(self._path):
            raise ValueError('Path does not exists: {}'.format(self._path))
        if self.is_root():
            i = 0
            refined_job_name = self._job_name
            while os.path.exists(os.path.join(self._path, refined_job_name)):
                refined_job_name = '{}_{}'.format(self._job_name, i)
                i += 1
            self._output_dir = os.path.join(self._path, refined_job_name)
            os.makedirs(self._output_dir)
            self._job_name = refined_job_name
        else:
            self._output_dir = os.path.join(self._path, self._job_name)
            assert os.path.exists(self._output_dir)
        with open(os.path.join(self._output_dir, '{}.log'.format(self._diary_id)), 'w') as f:
            f.write('Diary #{}, parent = {}\n'.format(self._diary_id, self._parent_id))

    def _close(self):
        if self.is_root():
            with open(os.path.join(self._output_dir, '{}.pkl'.format(self._job_name)), 'wb') as f:
                pickle.dump(self.descendant_iteration_data(), f)

    def _write(self, entries=None):
        with open(os.path.join(self._output_dir, '{}.log'.format(self._diary_id)), 'a') as f:
            for k, v in entries.items():
                f.write('{}: {}\n'.format(k, v))
            f.write('----------------\n')

    def _get_reader_config(self):
        return dict(reader_module=readers.MemoryReader.__module__,
                    reader_class=readers.MemoryReader.__name__,
                    reader_args=(None, os.path.join(self._output_dir, '{}.pkl'.format(self._job_name))))

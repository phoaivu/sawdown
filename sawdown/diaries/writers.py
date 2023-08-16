import collections
import os.path
import pickle
import sys


def from_proto(proto_diaries):
    _writers = []
    for diary in proto_diaries:
        field_name = diary.WhichOneof('diary')
        if field_name == 'memory_diary':
            _writers.append(MemoryWriter())
        elif field_name == 'stream_diary':
            _writers.append(StreamWriter(diary.stream_diary.stream))
        elif field_name == 'file_diary':
            _writers.append(FileWriter(diary.file_diary.path, diary.file_diary.job_name))
        elif field_name is not None:
            raise ValueError('Unsupported diary writer: {}'.format(field_name))
    return _writers


class DiaryWriter(object):
    def write(self, diary_id='', parent_id='', entries=None):
        raise NotImplementedError()

    def close(self):
        pass


class MemoryWriter(DiaryWriter):
    def __init__(self):
        self._iteration_data = collections.OrderedDict()

    def write(self, diary_id='', parent_id='', entries=None):
        diary_data = self._iteration_data.get(diary_id, None)
        if diary_data is None:
            diary_data = (parent_id, [])
            self._iteration_data[diary_id] = diary_data
        diary_data[1].append(entries)

    def get_iteration_data(self, diary_id=''):
        if diary_id in {'', None}:
            return self._iteration_data
        return self._iteration_data.get(diary_id, None)


class StreamWriter(DiaryWriter):

    def __init__(self, stream='stdout'):
        self._output_stream = sys.stdout if stream == 'stdout' else sys.stderr
        self._encountered = set()

    def write(self, diary_id='', parent_id='', entries=None):
        if diary_id not in self._encountered:
            self._encountered.add(diary_id)
            self._output_stream.write('Diary {}, parent={}\n'.format(diary_id, parent_id))
        prefix = '' if parent_id in {'', None} else '    '
        for k, v in entries.items():
            self._output_stream.write('{}{}: {}\n'.format(prefix, k, v))
        self._output_stream.write('{}---------------\n'.format(prefix))


class FileWriter(MemoryWriter):

    def __init__(self, path='.', job_name=''):
        MemoryWriter.__init__(self)
        self._path = os.path.abspath(path)
        i = 0
        refined_job_name = job_name
        while os.path.exists(os.path.join(self._path, refined_job_name)):
            i += 1
            refined_job_name = '{}_{}'.format(job_name, i)
        self._job_name = refined_job_name
        self._folder_path = os.path.join(self._path, self._job_name)
        self._files = {}

    def write(self, diary_id='', parent_id='', entries=None):
        MemoryWriter.write(self, diary_id, parent_id, entries)
        file_output = self._files.get(diary_id, None)
        if file_output is None:
            file_output = open(os.path.join(self._folder_path, '{}.log'.format(diary_id)), 'w')
            file_output.write('Diary {}, parent={}\n'.format(diary_id, parent_id))
            self._files[diary_id] = file_output
        for k, v in entries.items():
            file_output.write('{}: {}\n'.format(k, v))
        file_output.write('---------------\n')

    def close(self):
        MemoryWriter.close(self)
        with open(os.path.join(self._folder_path, '{}.pkl'.format(self._job_name)), 'wb') as f:
            pickle.dump(self._iteration_data, f)
        [f.close() for f in self._files]
        self._files.clear()

    def get_reader_config(self, diary_id=''):
        return dict(pickle_file=os.path.join(self._folder_path, '{}.pkl'.format(self._job_name)))

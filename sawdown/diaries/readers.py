import pickle

from sawdown.diaries import common


class MemoryReader(object):

    def __init__(self, records=None, pickle_file=''):
        if records is None and pickle_file == '':
            raise ValueError('At least one is non-empty.')
        self._records = records
        if pickle_file != '':
            with open(pickle_file, 'rb') as f:
                self._records = pickle.load(f)

    def _get_diary(self, diary_id=''):
        if self._records is None:
            raise ValueError('Iteration data is not yet available')
        if diary_id in {'', None}:
            # Find the root diary, which has no parent.
            diary_id = next((k for k, v in self._records.items() if v[0] in {'', None}), None)
        if diary_id is None or diary_id not in self._records:
            raise ValueError('Diary not found: {}'.format(diary_id))

        return self._records[diary_id][1]

    def iterations(self, _id='', keys=()):
        run_records = filter(lambda _entry: _entry.get('record_type', -1) == common.RecordTypes.ITERATION,
                             self._get_diary(_id))
        if len(keys) == 0:
            return run_records
        elif len(keys) == 1:
            return map(lambda entry: entry.get(keys[0], None), run_records)
        else:
            return map(lambda entry: tuple(entry.get(k, None) for k in keys), run_records)

    def solution(self, _id=None):
        solution = next((entry for entry in self._get_diary(_id)
                         if entry.get('record_type', -1) == common.RecordTypes.SOLUTION), None)
        if solution is not None:
            casted_solution = common.Solution()
            casted_solution.update(solution)
            solution = casted_solution
        return solution

import os
import multiprocessing
import queue

import numpy as np

import sawdown.diaries.async_diary
from sawdown import errors, diaries


class Worker(multiprocessing.Process):

    def __init__(self, problems, solutions, diary_messages, diary_response):
        multiprocessing.Process.__init__(self, name=self.__class__.__name__)
        self._problems = problems
        self._solutions = solutions
        self._diary_message = diary_messages
        self._diary_response = diary_response

    def run(self):
        problem, optimizer = self._problems.get()
        while problem is not None:
            try:
                optimizer._diary.set_queues(self._diary_message, self._diary_response)
                solution = optimizer.optimize()
            except RuntimeError as ex:
                solution = diaries.Solution(x=None, iteration=-1, objective=np.nan,
                                            termination=diaries.Termination.FAILED, exception=str(ex))
            self._solutions.put((problem, solution))
            problem, optimizer = self._problems.get()


class BranchAndBounder(diaries.AsyncDiaryMixIn):
    """
    Breadth-first multiprocess B&B
    """

    def __init__(self):
        diaries.AsyncDiaryMixIn.__init__(self)
        self._n_processes = os.cpu_count()

    def parallelize(self, n_processes=os.cpu_count()):
        if n_processes < 0:
            raise ValueError('Cannot solve your problem with negative parallelization')
        self._n_processes = n_processes
        return self

    def _initial_problems(self, diary):
        raise NotImplementedError()

    def _accept(self, solution, diary):
        raise NotImplementedError()

    def _branch(self, problem, solution, diary):
        raise NotImplementedError()

    def _sub_optimizer(self, problem, diary):
        """
        Returns an optimizer created for the given problem.
        :param problem:
        :return:
        """
        raise NotImplementedError()

    def _optimize(self, diary):
        best_objective = np.inf
        x_opt = None

        try:
            initial_problems = self._initial_problems(diary)
            assert len(initial_problems) > 0
        except errors.InitializationError as ex:
            diary.set_solution(x=None, objective=np.nan, termination=diaries.Termination.FAILED_INITIALIZATION,
                               reason=ex.reason)
            return diary.solution

        if self._n_processes == 0:
            problems = queue.Queue()
            solutions = queue.Queue()
        else:
            problems = multiprocessing.Queue()
            solutions = multiprocessing.Queue()
        n_unsolved_problems = 0

        for problem in initial_problems:
            assert isinstance(problem.sub_diary, sawdown.diaries.async_diary.AsyncDiary) \
                   and problem.sub_diary._response_queue is None \
                    and problem.sub_diary._message_queue is None
            problems.put((problem, self._sub_optimizer(problem, diary)))
        n_unsolved_problems = len(initial_problems)

        workers = [Worker(problems, solutions,
                          self._diary_message, self._diary_response) for _ in range(self._n_processes)]
        [p.start() for p in workers]

        for _ in diary.as_long_as(lambda: n_unsolved_problems > 0):
            if self._n_processes == 0:
                problem, optimizer = problems.get()
                try:
                    optimizer._diary.set_queues(self._diary_message, self._diary_response)
                    solution = optimizer.optimize()
                except RuntimeError as ex:
                    solution = diaries.Solution(x=None, iteration=-1, objective=np.nan,
                                                termination=diaries.Termination.FAILED,
                                                exception=str(ex))
                    solution.update(exception=str(ex))
                solutions.put((problem, solution))

            sub_problem, sub_solution = solutions.get()
            n_unsolved_problems -= 1

            if sub_solution.x is None:
                diary.set_items(sub_problem=sub_problem,
                                termination=sub_solution.termination,
                                msg_sub='Failed solving sub-problem: {}'.format(sub_solution.get('exception', '')))
            elif sub_solution.objective < best_objective:
                diary.set_items(x=sub_solution.x.copy(), objective=sub_solution.objective,
                                best_objective=best_objective)
                if self._accept(sub_solution, diary):
                    best_objective = sub_solution.objective
                    x_opt = sub_solution.x.copy()
                    diary.set_items(msg_sub='Better optima found. Updated best solution')
                else:
                    sub_problems = self._branch(sub_problem, sub_solution, diary)
                    n_unsolved_problems += len(sub_problems)
                    # [problems.put((p, self._sub_optimizer(p, diary))) for p in sub_problems]
                    for p in sub_problems:
                        optimizer = self._sub_optimizer(p, diary)
                        assert isinstance(p.sub_diary, sawdown.diaries.async_diary.AsyncDiary) \
                               and p.sub_diary._response_queue is None \
                               and p.sub_diary._message_queue is None
                        assert isinstance(optimizer._diary, sawdown.diaries.async_diary.AsyncDiary) \
                               and optimizer._diary._response_queue is None \
                               and optimizer._diary._message_queue is None
                        problems.put((p, optimizer))

                    diary.set_items(msg_sub='Promising optima found. '
                                            'Branch into {} sub-problems'.format(len(sub_problems)))
            else:
                diary.set_items(x=sub_solution.x.copy(), objective=sub_solution.objective,
                                best_objective=best_objective, msg_sub='Mediocre solution. Rejected.')

            if n_unsolved_problems == 0:
                diary.set_solution(x=x_opt, objective=best_objective, termination=diaries.Termination.MAX_ITERATION)
        [problems.put((None, None)) for _ in workers]
        [w.join() for w in workers]
        [w.close() for w in workers]
        workers.clear()
        return diary.solution

    def optimize(self):
        """
        Minimize the objective function with the given configuration.
        Return a `sawdown.diaries.Solution` object, from which the logs could be read via `solution.reader()`.

        :rtype: sawdown.diaries.Solution
        """
        diaries.AsyncDiaryMixIn._start_diary_worker(self)
        with self._diary:
            solution = self._optimize(self._diary)
        diaries.AsyncDiaryMixIn._stop_diary_worker(self)
        return solution

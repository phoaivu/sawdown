import multiprocessing
import queue
import importlib

import numpy as np

from sawdown import errors, diaries
from sawdown.optimizers import first_orders
from sawdown.proto import sawdown_pb2


def _solve(relaxed_problem, sub_problem, diary_message, diary_response, diary_response_semaphore):
    problem = sawdown_pb2.Problem()
    problem.CopyFrom(relaxed_problem)

    if sub_problem.HasField('initializer'):
        problem.ClearField('initializers')
        problem.initializers.append(sawdown_pb2.Initializer(fixed_initializer=sub_problem.initializer))
    for var in sub_problem.fixed_value_constraints:
        problem.fixed_value_constraints.append(var)
    for var in sub_problem.bound_constraints:
        problem.bound_constraints.append(var)

    try:
        optimizer = first_orders.AsyncFirstOrderOptimizer(
            problem, sub_problem.diary_id, diary_message, diary_response, diary_response_semaphore)
        solution = optimizer.optimize()
    except RuntimeError as ex:
        solution = diaries.Solution(x=None, iteration=-1, objective=np.nan,
                                    termination=diaries.Termination.FAILED, exception=str(ex))
    return solution


class Worker(multiprocessing.Process):

    def __init__(self, serialized_relaxed_problem, problems, solutions,
                 diary_messages, diary_response, response_semaphore):
        multiprocessing.Process.__init__(self, name=self.__class__.__name__)
        self._serialized_relaxed_problem = serialized_relaxed_problem
        self._problems = problems
        self._solutions = solutions
        self._diary_message = diary_messages
        self._diary_response = diary_response
        self._diary_response_semaphore = response_semaphore

    def run(self):
        # Since protobuf methods (in sawdown_pb2) are created when loaded, need to reload in a new process.
        importlib.reload(sawdown_pb2)

        relaxed_problem = sawdown_pb2.Problem()
        relaxed_problem.MergeFromString(self._serialized_relaxed_problem)

        _, problem_proto = self._problems.get()
        while problem_proto is not None:
            problem = sawdown_pb2.IntegerSubproblem()
            problem.MergeFromString(problem_proto)
            solution = _solve(relaxed_problem, problem,
                              self._diary_message, self._diary_response, self._diary_response_semaphore)
            self._solutions.put((problem_proto, solution))
            _, problem_proto = self._problems.get()


class BranchAndBounder(diaries.AsyncDiaryMixIn):
    """
    Breadth-first multiprocess B&B
    """

    def __init__(self, proto_problem):
        diaries.AsyncDiaryMixIn.__init__(self, proto_problem)
        self._proto_problem = proto_problem

    def _n_processes(self):
        return self._proto_problem.config.parallelization

    def _initial_problems(self, diary):
        raise NotImplementedError()

    def _accept(self, solution, diary):
        raise NotImplementedError()

    def _branch(self, sub_problem, sub_solution, diary):
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

        relaxed_problem = sawdown_pb2.Problem()
        relaxed_problem.CopyFrom(self._proto_problem)
        for field in ['diaries', 'integer_constraints', 'binary_constraints']:
            relaxed_problem.ClearField(field)

        if self._n_processes() == 0:
            problems = queue.PriorityQueue()
            solutions = queue.Queue()
        else:
            # TODO: multiprocessing PriorityQueue
            problems = multiprocessing.Queue()
            solutions = multiprocessing.Queue()
        n_unsolved_problems = 0

        for problem in initial_problems:
            problems.put((-np.inf, problem.SerializeToString()))
        n_unsolved_problems = len(initial_problems)

        workers = [Worker(relaxed_problem.SerializeToString(),
                          problems, solutions, self._diary_message, self._diary_response,
                          self._diary_response_semaphore) for _ in range(self._n_processes())]
        [p.start() for p in workers]

        for _ in diary.as_long_as(lambda: n_unsolved_problems > 0):
            if self._n_processes() == 0:
                _, problem_proto = problems.get()
                problem = sawdown_pb2.IntegerSubproblem()
                problem.MergeFromString(problem_proto)
                solution = _solve(relaxed_problem, problem,
                                  self._diary_message, self._diary_response, self._diary_response_semaphore)
                solutions.put((problem_proto, solution))

            sub_problem_proto, sub_solution = solutions.get()
            sub_problem = sawdown_pb2.IntegerSubproblem()
            sub_problem.MergeFromString(sub_problem_proto)
            n_unsolved_problems -= 1

            if sub_solution.x is None:
                diary.set_items(sub_problem=sub_problem,
                                termination=sub_solution.termination,
                                diary_id=sub_problem.diary_id,
                                msg_sub='Failed solving sub-problem: {}'.format(sub_solution.get('exception', '')))
            elif sub_solution.objective < best_objective:
                diary.set_items(x=sub_solution.x.copy(), objective=sub_solution.objective,
                                diary_id=sub_problem.diary_id,
                                best_objective=best_objective)
                if self._accept(sub_solution, diary):
                    best_objective = sub_solution.objective
                    x_opt = sub_solution.x.copy()
                    diary.set_items(msg_sub='Better optima found. Updated best solution')
                else:
                    sub_problems = self._branch(sub_problem, sub_solution, diary)
                    [problems.put((sub_solution.objective, p.SerializeToString())) for p in sub_problems]
                    n_unsolved_problems += len(sub_problems)
                    diary.set_items(msg_sub='Promising optima found. '
                                            'Branch into {} sub-problems'.format(len(sub_problems)))
            else:
                diary.set_items(x=sub_solution.x.copy(), objective=sub_solution.objective,
                                diary_id=sub_problem.diary_id,
                                best_objective=best_objective, msg_sub='Mediocre solution. Rejected.')

            if n_unsolved_problems == 0:
                diary.set_solution(x=x_opt, objective=best_objective, termination=diaries.Termination.MAX_ITERATION)
        [problems.put((0, None)) for _ in workers]
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
        try:
            with self._diary:
                return self._optimize(self._diary)
        finally:
            diaries.AsyncDiaryMixIn._stop_diary_worker(self)

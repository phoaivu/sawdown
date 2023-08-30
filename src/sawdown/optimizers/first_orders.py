import numpy as np

from sawdown import diaries, errors, constraints
from sawdown.optimizers import base


class FirstOrderOptimizerBase(base.OptimizerBase):

    def __init__(self, proto_problem):
        base.OptimizerBase.__init__(self, proto_problem)

    def __initialize(self, diary):
        initializer = None
        for init in self._initializers:
            initializer = init.initialize(initializer)

        with diary.sub() as sub_diary:
            sub_diary.set_items(msg='Starting initialization')
            initializer = self._constrained_initialize(initializer, self._config, self._opti_math, sub_diary)
        if initializer is None:
            raise errors.InitializationError('Failed to initialize. Missing initializer, maybe.')

        self._objective.check_dimensions(initializer.size)
        return initializer

    def __prestep(self, k, x_k, diary):
        return constraints.ConstraintsMixIn._prestep(self, k, x_k, self._opti_math, diary)

    def __direction(self, k, x_k, derivatives, diary):
        direction = self._direction_calculator.direction(k, x_k, derivatives, self._opti_math, diary)
        return constraints.ConstraintsMixIn._direction(self, k, x_k, direction, self._opti_math, diary)

    def __steplength(self, k, x_k, d_k, diary):
        steplength = 1.
        for calculator in self._steplength_calculators:
            steplength = calculator.steplength(k, x_k, d_k, steplength, self._opti_math)
        return constraints.ConstraintsMixIn._steplength(self, k, x_k, d_k, steplength, self._opti_math, diary)

    def __stop(self, k, x_k, delta, d_k):
        terminations = map(lambda stopper: stopper.stop(k, x_k, delta, d_k, self._opti_math), self._stoppers)
        return next((t for t in terminations if t != diaries.Termination.CONTINUE), diaries.Termination.CONTINUE)

    def _optimize(self, diary):
        """
        :return:
        """
        try:
            x_k = self.__initialize(diary)
        except errors.InitializationError as ex:
            diary.set_solution(None, objective=np.nan, termination=diaries.Termination.FAILED_INITIALIZATION,
                               msg=ex.reason, exception=str(ex))
            return diary.solution

        termination_type = diaries.Termination.CONTINUE
        for k in diary.as_long_as(lambda: termination_type == diaries.Termination.CONTINUE):
            x_k = self.__prestep(k, x_k, diary)
            objective = self._objective.objective(x_k)
            gradients = self._objective.deriv_variables(x_k)
            diary.set_items(objective=objective.copy(), derivative=gradients.copy())

            d_k = self.__direction(k, x_k, gradients, diary)
            delta = self.__steplength(k, x_k, d_k, diary)
            diary.set_items(d_k=d_k.copy(), delta=delta)

            x_k_prev = x_k.copy()
            x_k = x_k_prev + delta * d_k
            diary.set_items(x_k_prev=x_k.copy(), x_k_diff=x_k - x_k_prev, changes=delta * d_k, x_k=x_k.copy())

            termination_type = self.__stop(k, x_k, delta, d_k)
            if termination_type != diaries.Termination.CONTINUE:
                diary.set_solution(x=x_k.copy(), objective=self._objective.objective(x_k),
                                   termination=termination_type)
        return diary.solution


class FirstOrderOptimizer(FirstOrderOptimizerBase):

    def __init__(self, proto_problem):
        FirstOrderOptimizerBase.__init__(self, proto_problem)
        self._diary = diaries.SyncDiary(diary_id='', proto_diaries=proto_problem.diaries)

    def optimize(self):
        with self._diary:
            return self._optimize(self._diary)


class AsyncFirstOrderOptimizer(FirstOrderOptimizerBase):
    """
    First-order optimizer to be used in companion with branch-and-bound.
    The main difference is it runs with an asynchronous diary writer.
    """
    def __init__(self, proto_problem, diary_id, message_queue, response_queue, response_semaphore):
        FirstOrderOptimizerBase.__init__(self, proto_problem)
        self._diary = diaries.AsyncDiary(diary_id, message_queue, response_queue, response_semaphore)

    def optimize(self):
        with self._diary:
            return self._optimize(self._diary)

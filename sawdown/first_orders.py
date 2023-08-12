import numpy as np

from sawdown import opti_math, errors, objectives, initializers, constraints, \
    directions, steplengths, stoppers, diaries


class FirstOrderOptimizer(opti_math.OptiMathMixIn,
                          objectives.ObjectiveMixIn,
                          constraints.ConstraintsMixIn,
                          initializers.InitializerMixIn,
                          directions.DirectionMixIn, steplengths.SteplengthMixIn,
                          stoppers.StopperMixIn, diaries.DiaryMixIn):
    """
    Generic objective function (with a derivative function), with or without constraints.

    Optimizers are customized in chained calls.

    A first-order optimizer has the following components:
    - An objective function, replaced when chained.
    - A list of constraints, merged when chained.
    - A list of initializers, chain of responsibilities.
    - A direction calculator, replaced when chained.
    - A list of step-length calculators, chain of responsibilities.
    - A list of stoppers, optimization stops when any of the stoppers is satisfied.
    - A diary, replaced when chained.
    """

    def __init__(self):
        opti_math.OptiMathMixIn.__init__(self)
        objectives.ObjectiveMixIn.__init__(self)
        constraints.ConstraintsMixIn.__init__(self)
        initializers.InitializerMixIn.__init__(self)
        directions.DirectionMixIn.__init__(self)
        steplengths.SteplengthMixIn.__init__(self)
        stoppers.StopperMixIn.__init__(self)
        diaries.DiaryMixIn.__init__(self)

    def __initialize(self, diary):
        if self._objective is None:
            raise ValueError('Missing objective')
        if len(self._stoppers) == 0:
            raise ValueError('Missing stoppers: Machine Power is priceless, do not waste.')
        constraints.ConstraintsMixIn._setup(self, self._objective, self._opti_math)
        initializers.InitializerMixIn._setup(self, self._objective, self._opti_math)
        directions.DirectionMixIn._setup(self, self._objective, self._opti_math)
        steplengths.SteplengthMixIn._setup(self, self._objective, self._opti_math)
        stoppers.StopperMixIn._setup(self, self._objective, self._opti_math)
        diaries.DiaryMixIn._setup(self, self._objective, self._opti_math)

        initializer = initializers.InitializerMixIn._initialize(self)
        with diary.sub() as sub_diary:
            sub_diary.set_items(msg='Starting initialization')
            initializer = constraints.ConstraintsMixIn._initialize(self, initializer, sub_diary)
        if initializer is None:
            raise errors.InitializationError('Failed to initialize. Missing initializer, maybe.')

        self._objective.check_dimensions(initializer.size)
        return initializer

    def __prestep(self, k, x_k, diary):
        return constraints.ConstraintsMixIn._prestep(self, k, x_k, diary)

    def __direction(self, k, x_k, derivatives, diary):
        d_k = directions.DirectionMixIn._direction(self, k, x_k, derivatives, diary)
        return constraints.ConstraintsMixIn._direction(self, k, x_k, d_k, diary)

    def __steplength(self, k, x_k, d_k, diary):
        steplength = steplengths.SteplengthMixIn._steplength(self, k, x_k, d_k, diary)
        return constraints.ConstraintsMixIn._steplength(self, k, x_k, d_k, steplength, diary)

    def _optimize(self, diary):
        """
        :return:
        """
        try:
            x_k = self.__initialize(diary)
        except errors.InitializationError as ex:
            diary.set_solution(None, objective=np.nan, termination=diaries.Termination.FAILED_INITIALIZATION,
                               msg=ex.reason)
            return diary.solution

        termination_type = diaries.Termination.CONTINUE
        for k in diary.as_long_as(lambda: termination_type == diaries.Termination.CONTINUE):
            x_k = self.__prestep(k, x_k, diary)
            gradients = self._objective.deriv_variables(x_k)
            diary.set_items(derivative=gradients.copy())

            d_k = self.__direction(k, x_k, gradients, diary)
            delta = self.__steplength(k, x_k, d_k, diary)
            diary.set_items(x_k=x_k.copy(), d_k=d_k.copy(), delta=delta)

            x_k_prev = x_k.copy()
            # x_k = x_k_prev + delta * d_k
            x_k = np.add(x_k_prev, np.multiply(delta, d_k))
            diary.set_items(x_k_prev=x_k_prev, x_k_diff=x_k - x_k_prev,
                            changes=delta*d_k, x_k_dtype=x_k.dtype, x_k_prev_dtype=x_k_prev.dtype,
                            d_k_dtype=d_k.dtype)

            termination_type = self._stop(k, x_k, delta, d_k)
            if termination_type != diaries.Termination.CONTINUE:
                diary.set_solution(x=x_k.copy(), objective=self._objective.objective(x_k),
                                   termination=termination_type)

        return diary.solution

    def optimize(self, diary=None):
        """
        Minimize the objective function with the given configuration.

        When `diary` is None, the optimizer is responsible for resource clean-up,
         including reset the diary when called repeatedly.

        When `diary` is not None, the caller is responsible for resource clean-up
        (the recommended way is via a `with` block).

        Return a `sawdown.diaries.Solution` object, from which the logs could be read via `solution.reader()`.

        :param diary:
        :rtype: sawdown.diaries.Solution
        """
        if diary is not None:
            return self._optimize(diary)

        diary = self._diary if self._diary is not None else diaries.log_stream('')
        with diary:
            return self._optimize(diary)

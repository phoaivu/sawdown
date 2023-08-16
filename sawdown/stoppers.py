import numpy as np
from sawdown import diaries


class StopperBase(object):
    def setup(self, objective, opti_math, **kwargs):
        pass


class MaxIterationsStopper(StopperBase):
    def __init__(self, max_iters=1000):
        self._max_iters = max_iters

    def stop(self, k, x_k, delta, d_k):
        return diaries.Termination.CONTINUE if k < self._max_iters else diaries.Termination.MAX_ITERATION

    def clone(self):
        return MaxIterationsStopper(self._max_iters)


class InfinitesimalStepStopper(StopperBase):
    def __init__(self):
        self._opti_math = None

    def setup(self, objective, opti_math, **kwargs):
        self._opti_math = opti_math
        if self._opti_math is None:
            raise ValueError('Empty configuration.')

    def stop(self, k, x_k, delta, d_k):
        if self._opti_math.true_leq(np.max(np.abs(delta * d_k) / np.maximum(np.abs(x_k), 1.)),
                                    np.power(self._opti_math.epsilon, 2. / 3)):
            return diaries.Termination.INFINITESIMAL_STEP
        return diaries.Termination.CONTINUE

    def clone(self):
        return InfinitesimalStepStopper()

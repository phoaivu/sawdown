import numpy as np
from sawdown import diaries


class StopperBase(object):
    """
    All hails Donald Knuth.
    """
    pass


class MaxIterationsStopper(StopperBase):
    def __init__(self, max_iters=1000):
        self._max_iters = max_iters

    def stop(self, k, x_k, delta, d_k, opti_math):
        return diaries.Termination.CONTINUE if k < self._max_iters else diaries.Termination.MAX_ITERATION


class InfinitesimalStepStopper(StopperBase):
    def __init__(self):
        pass

    def stop(self, k, x_k, delta, d_k, opti_math):
        if opti_math.true_leq(np.max(np.abs(delta * d_k) / np.maximum(np.abs(x_k), 1.)),
                                    np.power(opti_math.epsilon, 2. / 3)):
            return diaries.Termination.INFINITESIMAL_STEP
        return diaries.Termination.CONTINUE

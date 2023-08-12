import numpy as np
from sawdown import diaries


class StopperMixIn(object):
    def __init__(self):
        self._stoppers = []

    def _setup(self, objective, opti_math, **kwargs):
        [s.setup(objective, opti_math, **kwargs) for s in self._stoppers]

    def _stop(self, k, x_k, delta, d_k):
        terminations = map(lambda stopper: stopper.stop(k, x_k, delta, d_k), self._stoppers)
        return next((t for t in terminations if t != diaries.Termination.CONTINUE), diaries.Termination.CONTINUE)

    def stoppers_from(self, other):
        assert isinstance(other, StopperMixIn)
        self._stoppers = [s.clone() for s in other._stoppers]
        return self

    def stop_after(self, max_iters=1000):
        self._stoppers.append(MaxIterationsStopper(max_iters=max_iters))
        return self

    def stop_small_steps(self):
        self._stoppers.append(InfinitesimalStepStopper())
        return self


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

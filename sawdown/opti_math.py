"""
Replacement of elementary operations (mostly comparisons) with a relaxed precision.
"""
import numpy as np

from sawdown import errors, diaries


class SawMath(object):
    """
    All operations take numpy arrays of floats and return element-wise results.
    Wrote for finite floats.
    """

    def __init__(self, epsilon=1e-24):
        self._epsilon = epsilon
        self._sqrt_epsilon = np.sqrt(epsilon)

    @staticmethod
    def _true_negative(x):
        # This is to take into account np.signbit(-0.) = True.
        return np.signbit(np.minimum(x, 0.))

    @staticmethod
    def true_leq(x, y):
        return np.logical_not(SawMath._true_negative(y - x))

    def equal_zeros(self, x):
        return np.logical_not(self.non_zeros(x))

    def non_zeros(self, x):
        return SawMath._true_negative(self._epsilon - np.square(x))

    def true_negative(self, x):
        return SawMath._true_negative(x)

    def negative(self, x):
        return SawMath._true_negative(x + self._sqrt_epsilon)

    def positive(self, x):
        return SawMath._true_negative(self._sqrt_epsilon - x)

    def non_positive(self, x):
        return np.logical_not(self.positive(x))

    def non_negative(self, x):
        return np.logical_not(self.negative(x))

    def in_bounds(self, x, lower, upper):
        return np.logical_and(self.non_positive(x - upper), self.non_negative(x - lower))

    def equals(self, x, y):
        return self.equal_zeros(x - y)

    def lt(self, x, y):
        return self.negative(x - y)

    def gt(self, x, y):
        return self.positive(x - y)

    def leq(self, x, y):
        return self.non_positive(x - y)

    def geq(self, x, y):
        return self.non_negative(x - y)


class OptiMath(SawMath):
    """
    Maths and configurations needed for optimization.
    """
    def __init__(self, epsilon=1e-24, initialization_max_iters=500, initialization_decay_steps=50):
        SawMath.__init__(self, epsilon)
        self.initialization_max_iters = initialization_max_iters
        self.initialization_decay_steps = initialization_decay_steps

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if not self._true_negative(-value):
            raise ValueError('Invalid value for epsilon: {}'.format(value))

        self._epsilon = value
        self._sqrt_epsilon = np.sqrt(self._epsilon)

    def config(self, **kwargs):
        names = {'epsilon', 'initialization_max_iters', 'initialization_decay_steps'}
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)
            else:
                raise ValueError('Unknown math config: {}'.format(k))

    def optimize(self, initializer, satisfier, director, stepper, diary):
        x_k = initializer.copy()
        if satisfier(x_k):
            diary.set_solution(x=x_k.copy(), objective=np.nan, termination=diaries.Termination.SATISFIED)
            return x_k

        termination = diaries.Termination.CONTINUE
        for k in diary.as_long_as(lambda: termination == diaries.Termination.CONTINUE):
            d_k = director(x_k, diary)
            delta = stepper(k, x_k, d_k, diary)
            diary.set_items(x_k=x_k.copy(), delta=delta, d_k=d_k.copy())

            x_k += delta * d_k

            if satisfier(x_k):
                termination = diaries.Termination.SATISFIED
            elif self.true_leq(np.max(np.abs(delta * d_k) / np.maximum(np.abs(x_k), 1.)),
                               np.power(self._epsilon, 2. / 3)):
                termination = diaries.Termination.INFINITESIMAL_STEP
            elif k >= self.initialization_max_iters:
                termination = diaries.Termination.MAX_ITERATION

        diary.set_solution(x=x_k.copy(), objective=np.nan, termination=termination)
        if not satisfier(x_k):
            raise errors.InitializationError('Unsatisfied initialization after {} iterations'.format(
                diary.solution.iteration))
        return x_k


class OptiMathMixIn(object):

    def __init__(self):
        self._opti_math = OptiMath()

    def config(self, **kwargs):
        self._opti_math.config(**kwargs)
        return self

    def epsilon(self, epsilon=1e-24):
        """
        Machine precision, used mainly in stop conditions and initialization.

        :param epsilon:
        :return:
        """
        return self.config(epsilon=epsilon)

    def initialization_max_iters(self, max_iters=500):
        return self.config(initialization_max_iters=max_iters)

    def initialization_decay_steps(self, decay_steps=20):
        return self.config(initialization_decay_steps=decay_steps)

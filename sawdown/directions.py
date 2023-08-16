import numpy as np


class DirectionCalculatorBase(object):

    def clone(self):
        raise NotImplementedError()

    def setup(self, objective, opti_math, **kwargs):
        pass

    def direction(self, k, x_k, derivatives, diary):
        raise NotImplementedError()


class SteepestDecent(DirectionCalculatorBase):
    def __init__(self):
        pass

    def clone(self):
        return SteepestDecent()

    def direction(self, k, x_k, derivatives, diary):
        return -derivatives


class ConjugateGradient(DirectionCalculatorBase):
    """
    See p.121, "Numerical Optimization". May not work.
    """

    def __init__(self, beta=0.9):
        self._previous_deriv = None
        self._beta = beta

    def clone(self):
        return ConjugateGradient()

    def direction(self, k, x_k, derivatives, diary):
        conjugated_deriv = derivatives.copy()
        if self._previous_deriv is not None:
            conjugated_deriv = ((1. - self._beta) * self._previous_deriv) + (self._beta * conjugated_deriv)
            # diary.set_items(beta=beta, msg=self.__class__.__name__)
        self._previous_deriv = conjugated_deriv.copy()
        return -conjugated_deriv


class Adam(DirectionCalculatorBase):

    def __init__(self, alpha=0.9, beta=0.999):
        self._alpha = alpha
        self._beta = beta
        self._first_moment = None
        self._second_moment = None
        self._opti_math = None

    def clone(self):
        return Adam(self._alpha, self._beta)

    def setup(self, objective, opti_math, **kwargs):
        self._opti_math = opti_math

    def direction(self, k, x_k, derivatives, diary):
        if self._first_moment is None:
            self._first_moment = np.zeros_like(derivatives, dtype=float)
            self._second_moment = np.zeros_like(derivatives, dtype=float)
        self._first_moment = (1. - self._alpha) * self._first_moment + (self._alpha * derivatives)
        self._second_moment = (1. - self._beta) * self._second_moment + (self._beta * np.square(derivatives))
        coeff = np.sqrt(1. - np.power(self._beta, k + 1)) / (1. - np.power(self._alpha, k + 1))
        return -coeff * (self._first_moment / (np.sqrt(self._second_moment) + self._opti_math.epsilon))

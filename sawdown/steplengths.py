import numpy as np


# TODO: decayed steplength is Computer Scientist's trick. Maybe replace it by Wolfe conditions.


class SteplengthBase(object):
    def setup(self, objective, opti_math, **kwargs):
        pass

    def steplength(self, k, x_k, d_k, max_steplength):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()


class DecaySteplength(SteplengthBase):

    def __init__(self, decay_steps=100):
        SteplengthBase.__init__(self)
        self._decay_steps = decay_steps

    def steplength(self, k, x_k, d_k, max_steplength):
        return max_steplength * np.exp(-float(k) / self._decay_steps)

    def clone(self):
        return DecaySteplength(self._decay_steps)


class QuadraticInterpolationSteplength(SteplengthBase):
    def __init__(self):
        SteplengthBase.__init__(self)
        self._objective = None

    def setup(self, objective, opti_math, **kwargs):
        self._objective = objective

    def steplength(self, k, x_k, d_k, max_steplength):
        """

        :param k:
        :param x_k:
        :param d_k:
        :param max_steplength:
        :return:
        """
        interpolated_step = 0.
        b = np.matmul(self._objective.deriv_variables(x_k).T, d_k)
        a = self._objective.objective(x_k + d_k) - self._objective.objective(x_k) - b
        if a * b < 0.:
            interpolated_step = min((-b / (2. * a)).squeeze(), max_steplength)
        return interpolated_step

    def clone(self):
        return QuadraticInterpolationSteplength()


class CircleDetectionSteplength(SteplengthBase):
    """
    If detect x_k loops over a list of positions, then halves the step-length.
    """
    def __init__(self, circle_length=2):
        SteplengthBase.__init__(self)
        self._circle_length = circle_length
        self._position_history = None
        self._opti_math = None

    def setup(self, objective, opti_math, **kwargs):
        self._opti_math = opti_math

    def steplength(self, k, x_k, d_k, max_steplength):
        if self._position_history is None:
            self._position_history = x_k[:, None].copy()
        elif self._position_history.shape[1] < (self._circle_length * 2):
            self._position_history = np.hstack((self._position_history, x_k[:, None]))
        else:
            self._position_history[:, :-1] = self._position_history[:, 1:]
            self._position_history[:, -1:] = x_k[:, None]

            if np.all(self._opti_math.equals(self._position_history[:, :self._circle_length],
                                             self._position_history[:, self._circle_length:])):
                max_steplength /= 2.
        return max_steplength

    def clone(self):
        return CircleDetectionSteplength(self._circle_length)

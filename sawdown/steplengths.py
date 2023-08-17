import numpy as np


class SteplengthBase(object):
    def setup(self, objective, **kwargs):
        pass

    def steplength(self, k, x_k, d_k, max_steplength, opti_math):
        raise NotImplementedError()

    def clone(self):
        raise NotImplementedError()


class DecaySteplength(SteplengthBase):

    def __init__(self, decay_steps=100):
        SteplengthBase.__init__(self)
        self._decay_steps = decay_steps

    def steplength(self, k, x_k, d_k, max_steplength, opti_math):
        return max_steplength * np.exp(-float(k) / self._decay_steps)

    def clone(self):
        return DecaySteplength(self._decay_steps)


class QuadraticInterpolationSteplength(SteplengthBase):
    def __init__(self):
        SteplengthBase.__init__(self)
        self._objective = None

    def setup(self, objective, **kwargs):
        self._objective = objective

    def steplength(self, k, x_k, d_k, max_steplength, opti_math):
        """

        :param k:
        :param x_k:
        :param d_k:
        :param max_steplength:
        :param opti_math:
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
    def __init__(self, circle_length=2, decay_rate=0.5):
        SteplengthBase.__init__(self)
        self._circle_length = circle_length
        self._position_history = None
        self._decay_rate = decay_rate
        self._multiplier = 1.

    def steplength(self, k, x_k, d_k, max_steplength, opti_math):
        if self._position_history is None:
            self._position_history = x_k[:, None].copy()
        elif self._position_history.shape[1] < (self._circle_length * 2):
            self._position_history = np.hstack((self._position_history, x_k[:, None]))
        else:
            self._position_history[:, :-1] = self._position_history[:, 1:]
            self._position_history[:, -1:] = x_k[:, None]

            if np.all(opti_math.equals(self._position_history[:, :self._circle_length],
                                       self._position_history[:, self._circle_length:])):
                self._multiplier *= self._decay_rate
        return self._multiplier * max_steplength

    def clone(self):
        return CircleDetectionSteplength(self._circle_length)

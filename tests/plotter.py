import numpy as np
from matplotlib import pyplot as plt


class Plotter(object):
    """
    Mostly for debugging toy examples.
    """

    def __init__(self, optimizer, reader):
        self._optimizer = optimizer
        self._reader = reader

    def draw(self, figsize=(10, 8), resolution=5000, from_iter=0, to_iter=None, show=True):
        xlim = (-10, 10)
        ylim = (-10, 10)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, box_aspect=1.)
        ax.set_xlim(*xlim)
        ax.grid(True)

        # step = (ylim[1] - ylim[0]) / float(resolution)
        xgrid, ygrid = np.meshgrid(np.arange(xlim[0], xlim[1], (xlim[1] - xlim[0]) / float(resolution)),
                                   np.arange(ylim[0], ylim[1], (ylim[1] - ylim[0]) / float(resolution)))
        data = np.vstack((xgrid.flatten(order='F'), ygrid.flatten(order='F')))

        objective_contour = self._optimizer.objective(data)
        ax.contour(xgrid, ygrid, objective_contour.reshape(xgrid.shape, order='F'), levels=50, alpha=0.5)

        constraint_contour = self._optimizer.residuals(data)
        for i in range(constraint_contour.shape[0]):
            ax.contour(xgrid, ygrid, constraint_contour[i, :].reshape(xgrid.shape, order='F'), levels=[0], alpha=1.)

        x_k = np.asarray(list(self._reader.iterations(_id='', keys=('x_k', ))), dtype=float).T
        # d_k = np.asarray(list(self._logger.get_history('d_k')), dtype=float).T

        ax.plot(x_k[0, :], x_k[1, :], 'b.')
        adjustment = 0.5
        to_iter = to_iter if to_iter is not None else np.inf
        from_iter = max(0, from_iter) if from_iter is not None else 0
        for k, (x, d, deriv, delta, objective) in enumerate(self._reader.iterations(_id='',
                keys=('x_k', 'd_k', 'derivative', 'delta', 'objective'))):
            if from_iter <= k <= to_iter:
                if objective is not None:
                    ax.text(x[0] + adjustment, x[1] - adjustment, '{:.6f}'.format(objective), fontsize=12)
                if deriv is not None:
                    ax.arrow(x[0], x[1], -deriv[0], -deriv[1], width=.003, linestyle='--', head_width=0.1,
                             facecolor='b', edgecolor='b', length_includes_head=True)
                if d is not None:
                    ax.arrow(x[0], x[1], delta * d[0], delta * d[1], width=.001, linestyle='-', head_width=0.2,
                             facecolor='r', edgecolor='r', length_includes_head=True, shape='right')
            if k > to_iter:
                break

        if show:
            plt.show()


class ObjectivePlotter(object):
    def __init__(self, logger):
        self._logger = logger

    def draw(self, show=True, title=''):
        fig = plt.figure(figsize=(6, 4))
        plt.title(title)
        ax = fig.add_subplot(311)
        ax.grid(True)
        ax.plot(list(self._logger.get_history('objective')), 'b')
        ax.set_ylabel('Objective')

        ax = fig.add_subplot(312)
        ax.grid(True)
        derivs = map(lambda x: np.sum(np.square(x)), self._logger.get_history('d_k'))
        ax.plot(list(derivs), 'r')
        ax.set_ylabel('Length of d_k')

        ax = fig.add_subplot(313)
        ax.grid(True)
        ax.plot(list(self._logger.get_history('delta')), 'black')
        ax.set_ylabel('Steplength')

        if show:
            plt.show()

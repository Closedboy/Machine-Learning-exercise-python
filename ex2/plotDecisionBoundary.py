import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y, legend, label, title=None):
    plt.ion()
    plotData(X[:, 1:], y, legend, label)
    n = X.shape[1]
    if n <= 3:
        plot_x = np.array([X[:, 1].min(axis=0) - 2, X[:, 1].max(axis=0) + 2])
        plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])
        plt.plot(plot_x, plot_y, label='Decision Boundary')
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros([len(u), len(v)])
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeature(u[i], v[j], 1) @ theta
        plt.contour(u, v, z.T, [0], linewidths=2)
    if title:
        plt.title(title)
    plt.legend()
    plt.ioff()
    plt.show()
    plt.close('all')



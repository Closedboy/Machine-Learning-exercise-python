import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y, legend, label):
    fig = plt.figure(figsize=(10, 8), dpi=70)
    pos_X = X[np.where(y == 1)]
    neg_X = X[np.where(y == 0)]
    plt.scatter(pos_X[:, 0], pos_X[:, 1], c='b', marker='+', linewidths=1, label=legend[0])
    plt.scatter(neg_X[:, 0], neg_X[:, 1], c='y', marker='o', linewidths=1, label=legend[1])
    plt.title('data')
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    plt.legend()
    return fig

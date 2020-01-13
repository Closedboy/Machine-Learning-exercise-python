import numpy as np


def load_data(path):
    data = np.loadtxt(path, dtype=float, delimiter=',')
    return data

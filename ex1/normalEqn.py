import numpy as np


def normalEqn(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

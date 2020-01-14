import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y, requires_grad=False):
    epsilon = 1e-5
    m = y.size
    h = sigmoid(X @ theta)
    J = -1.0 / m * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    if requires_grad:
        grad = 1.0 / m * X.T @ (h - y)
        return J, grad
    else:
        return J

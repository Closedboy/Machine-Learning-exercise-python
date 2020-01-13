import numpy as np
from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iterations):
    m = X.shape[0]
    J_history = []
    for i in range(iterations):
        J_history.append(computeCost(X, y, theta))
        d_J = 1.0 / m * X.T @ (X @ theta - y)
        theta = theta - alpha * d_J
    return theta, np.array(J_history).reshape(iterations, )

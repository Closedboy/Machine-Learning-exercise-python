import numpy as np
from costFunction import costFunction


def gradientDescent(X, y, theta, alpha, iterations):
    J_history = []
    for i in range(iterations):
        J, grad = costFunction(theta, X, y, requires_grad=True)
        J_history.append(J)
        theta = theta - alpha * grad
    return theta, np.array(J_history).reshape(iterations, )

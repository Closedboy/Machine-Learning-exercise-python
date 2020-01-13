def computeCost(x, y, theta):
    m = x.shape[0]
    h = x @ theta
    return 1.0 / (2.0 * m) * (h - y).T @ (h - y)

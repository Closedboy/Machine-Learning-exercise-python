# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: costFunctionReg.py
@Time    : 20-1-10 下午1:27
"""
import numpy as np
from sigmoid import sigmoid


def costFunctionReg(theta, X, y, reg_factor, requires_grad=False):
    epsilon = 1e-5
    m = np.size(X, axis=0)
    h = sigmoid(X @ theta)
    J = -1.0 / m * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon)) \
        + reg_factor / (2.0 * m) * theta[1:].T @ theta[1:]
    if requires_grad:
        grad = 1.0 / m * X.T @ (h - y)
        grad[1:] = grad[1:] + reg_factor / m * theta[1:]
        return J, grad
    else:
        return J


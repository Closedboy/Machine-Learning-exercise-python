# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: lr_gradient.py
@Time    : 20-1-14 下午9:04
"""
import numpy as np
from sigmoid import sigmoid


def lr_gradient(theta, X, y, reg_factor):
    m = len(y)
    h = sigmoid(X @ theta)
    grad = np.dot(X.T, h - y) / m
    grad[1:] = grad[1:] + reg_factor * theta[1:] / m
    return grad

# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: mapFeature.py
@Time    : 20-1-10 下午1:08
"""
import numpy as np


def mapFeature(X1, X2, m):
    """
    MAPFEATURE(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.
    Inputs X1, X2 must be the same size
    :param X1: original feature x1
    :param X2: original feature x2
    :param m: numbers of X1's rows
    :return: a new feature array with more features, comprising of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    degree = 6
    new_feature = 1.0 if m == 1 else np.ones([m, 1])
    for i in range(1, degree + 1):
        for j in range(i + 1):
            new_feature = np.hstack((new_feature, np.power(X1, i - j) * np.power(X2, j)))
    return new_feature

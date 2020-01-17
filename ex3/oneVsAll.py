# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: oneVsAll.py
@Time    : 20-1-14 下午7:39
"""
import numpy as np
import scipy.optimize as optim
from costFunctionReg import costFunctionReg
from lr_gradient import lr_gradient


def oneVsAll(X, y, num_labels, reg_factor):
    m, n = X.shape
    all_theta = np.zeros([num_labels, n])
    initial_theta = np.zeros(n)
    for i in range(num_labels):
        print('===========Training {}th classifier...===========\n'.format(i + 1))
        new_y = np.where(y == i, 1, 0)
        theta = optim.fmin_cg(costFunctionReg, initial_theta, lr_gradient,
                              args=(X, new_y, reg_factor), disp=True, maxiter=50)
        all_theta[i, :] = theta
    return all_theta

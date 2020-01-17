# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: ex3.py
@Time    : 20-1-14 上午10:45

Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
Instructions
------------
This file contains code that helps you get started on the linear exercise.
Following functions are implemented:

"""
import numpy as np
from scipy.io import loadmat
from displayData import displayData
from costFunctionReg import costFunctionReg
from oneVsAll import oneVsAll
from sigmoid import sigmoid


# =========== Part 1: Loading and Visualizing Data =============
print('===========Loading and Visualizing Data ...===========\n')
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']
displayData(np.copy(X), image_size=(20, 20), disp_rows=10, disp_cols=10, padding=1, shuffle=True)

y = y.reshape(-1, )
y[np.where(y == 10)] = 0
m = len(y)
X = np.concatenate((np.ones([m, 1]), X), axis=1)
input("Program paused. Press enter to continue.\n")

# ============ Part 2a: Vectorize Logistic Regression ============
print('===========Testing CostFunctionReg with regularization===========')
test_theta = np.array([-2, -1, 1, 2])
X_test = np.concatenate((np.ones([5, 1]), np.arange(1, 16, dtype=float).reshape(5, 3, order='F') / 10), axis=1)
y_test = np.array([1, 0, 1, 0, 1])
reg_factor_test = 3

J, grad = costFunctionReg(test_theta, X_test, y_test, reg_factor_test, requires_grad=True)
print('Cost: %f' % J)
print('Expected cost: 2.534819\n')
print('Gradients:\n', grad)
print('Expected gradients:\n0.146561\n -0.548558\n 0.724722\n 1.398003\n')
input("Program paused. Press enter to continue.\n")

# ============ Part 2b: One-vs-All Training ============
print('===========Training One-vs-All Logistic Regression...===========\n')

reg_factor = 0.1
num_labels = 10
all_theta = oneVsAll(X, y, num_labels, reg_factor)
input("Program paused. Press enter to continue.\n")

# ================ Part 3: Predict for One-Vs-All ================

h = sigmoid(X @ all_theta.T)
pre = np.argmax(h, axis=1)
accuracy = np.mean((pre == y).astype(np.float)) * 100
print('Training Set Accuracy: %f' % accuracy)



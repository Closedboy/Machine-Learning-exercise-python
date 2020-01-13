# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: ex2_reg.py
@Time    : 20-1-10 上午11:27

Machine Learning Online Class - Exercise 2: Logistic Regression
Instructions
------------
This file contains code that helps you get started on second part of the exercise
which covers regularization with logistic regression.
Following functions are implemented:
    sigmoid.py
    mapFeature.py
    costFunctionReg.py
    plotData.py
    plotDecisionBoundary.py

Data:
    the first column of x refers to Microchip Test 1
    the second column of x refers to Microchip Test 2
    y refers to whether pass the quality assurance or not [0, 1]
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
from plotData import plotData
from sigmoid import sigmoid
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from plotDecisionBoundary import plotDecisionBoundary


# Load Data
data = np.loadtxt('./ex2data2.txt', dtype=float, delimiter=',')
X, y = data[:, :2], data[:, 2]
np.set_printoptions(precision=4, suppress=True)

fig = plotData(X, y, legend=['y=1', 'y=0'], label=['Microchip Test 1', 'Microchip Test 2'])
plt.show()
plt.close(fig)

# =========== Part 1: Regularized Logistic Regression ============
m = y.shape[0]
X = mapFeature(X[:, 0, np.newaxis], X[:, 1, np.newaxis], m)
n = X.shape[1]
initial_theta = np.zeros([n])

# Set regularization parameter lambda to 1
reg_factor = 1.0

# Compute and display initial cost and gradient for regularized logistic regression
cost, grad = costFunctionReg(initial_theta, X, y, reg_factor, requires_grad=True)
print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n', grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085, 0.0188, 0.0001, 0.0503, 0.0115\n')
input("Program paused. Press enter to continue.\n\n")

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = np.ones([n], dtype=float)
cost, grad = costFunctionReg(test_theta, X, y, reg_factor=10.0, requires_grad=True)
print('Cost at test theta (with lambda = 10): %f' % cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at initial theta (zeros) - first five values only:\n', grad[:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460, 0.1614, 0.1948, 0.2269, 0.0922\n')
input("Program paused. Press enter to continue.\n")

# ============= Part 2: Regularization and Accuracies =============
reg_factor = [0, 1, 10, 100]

for r in reg_factor:
    res = optim.minimize(costFunctionReg, initial_theta, (X, y, r), options={'maxiter': 300, 'disp': False})
    theta = res.x
    cost = res.fun

    plotDecisionBoundary(theta, X, y, legend=['y=1', 'y=0'],
                         label=['Microchip Test 1', 'Microchip Test 2'],
                         title='Decision Boundary at lamda = {}'.format(r))

    p = sigmoid(X @ theta)
    pre = np.where(p >= 0.5, 1, 0)
    accuracy = np.mean((pre == y).astype(np.float)) * 100
    print('Train Accuracy with lambda = {}: {}'.format(r, accuracy))


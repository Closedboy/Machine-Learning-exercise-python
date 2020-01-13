"""
@Author: Zijian Huang
@Contact: zijian_huang@sjtu.edu.cn
@File: ex1.py
@Time: 2020/1/09 8:21 PM

Machine Learning Online Class - Exercise 1: Linear regression with multiple variables
Instructions
------------
This file contains code that helps you get started on the linear exercise.
Following functions are implemented:
    conputeCost.py
    gradientDescent.py
    featureNormalize.py
    normalEqn.py

Data:
    first column of x refers to the size of the house (in square feet)
    second column of x refers to the number of bedrooms
    y refers to the price of house
"""


import numpy as np
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
from normalEqn import normalEqn


# ================ Part 1: Feature Normalization ================
print('================Loading data ...================\n')
data = np.loadtxt('./ex1data2.txt', dtype=float, delimiter=',')
X, y = data[:, :2], data[:, 2, np.newaxis]
m = y.shape[0]
np.set_printoptions(precision=3, suppress=True)

print('First 10 examples from the dataset: \n')
print('x =\n', X[:10, :])
print('y =\n', y[:10])

input("Program paused. Press enter to continue.\n")


# Scale features and set them to zero mean
print('Normalizing Features ...\n')
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones([m, 1], dtype=float), X), axis=1)

# ================ Part 2: Gradient Descent ================
print('================Running gradient descent ...================\n')

# Hyperparameters of gradient descend
iterations = 50
alpha = 1

theta = np.zeros([3, 1])  # initialize fitting parameters
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)

# Plot the convergence graph
fig = plt.figure(figsize=(10, 8), dpi=70)
plt.plot(J_history, '-b', linewidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence graph of cost J')
plt.show()
plt.close(fig)

# Display gradient descent's result
print('Theta computed from gradient descent: \n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house
X_orig = np.array([[1650, 3]])
X_orig = (X_orig - mu) / sigma
price = np.concatenate((np.array([[1.0]]), X_orig), axis=1) @ theta

print('Predicted price of a 1650 sq-ft, 3 br house(using gradient descent): $%f\n' % price)

input("Program paused. Press enter to continue.\n")

# ================ Part 3: Normal Equations ================
print('================Solving with normal equations...================\n')
data = np.loadtxt('./ex1data2.txt', dtype=float, delimiter=',')
X, y = data[:, :2], data[:, 2, np.newaxis]
m = y.shape[0]
X = np.concatenate((np.ones([m, 1], dtype=float), X), axis=1)
theta = normalEqn(X, y)
print('Theta computed from the normal equations: \n', theta)

# Estimate the price of a 1650 sq-ft, 3 br house
X_orig = np.array([[1, 1650, 3]])
price = X_orig @ theta
print('Predicted price of a 1650 sq-ft, 3 br house(using normal equations): $%f\n' % price)


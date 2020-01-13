"""
@Author: Zijian Huang
@Contact: zijian_huang@sjtu.edu.cn
@File: ex1.py
@Time: 2020/1/08 8:21 PM

Machine Learning Online Class - Exercise 1: Linear Regression
Instructions
------------
This file contains code that helps you get started on the linear exercise.
Following functions are implemented:
    conputeCost.py
    gradientDescent.py

Data:
    x refers to the population size in 10,000s
    y refers to the profit in $10,000s
"""


import numpy as np
from computeCost import computeCost
from gradientDescent import gradientDescent
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


# ======================= Part 1: Plotting =======================
print('=======================Plotting Data ...=======================\n')
data = np.loadtxt('./ex1data1.txt', dtype=float, delimiter=',')
X, y = data[:, 0, np.newaxis], data[:, 1, np.newaxis]
np.set_printoptions(precision=3, suppress=True)

# plotData(X, y)
plt.ion()
fig = plt.figure(figsize=(10, 8), dpi=70)
plt.plot(X, y, 'ro', label='Training data')
plt.title('data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')

input("Program paused. Press enter to continue.\n")

# =================== Part 2: Cost and Gradient descent ===================
m = X.shape[0]  # number of training examples
X = np.concatenate((np.ones([m, 1]), X), axis=-1)  # Add a column of ones to X
theta = np.zeros([2, 1])  # initialize fitting parameters

# Hyperparameters of gradient descend
iterations = 1500
alpha = 0.01

# compute and display initial cost
J = computeCost(X, y, theta)
print('with theta = [0 ; 0]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 54.24\n')

input("Program paused. Press enter to continue.\n")

print('\n===================Running Gradient Descent ...===================\n')
# run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n', theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

# Plot the linear fit
plt.plot(X[:, 1], X @ theta, '-', label='Linear regression')
plt.legend()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([[1.0, 3.5]]) @ theta
print('For population = 35,000, we predict a profit of %f\n' % (predict1 * 10000))
predict2 = np.array([[1, 7]]) @ theta
print('For population = 70,000, we predict a profit of %f\n' % (predict2 * 10000))
input("Program paused. Press enter to continue.\n")

# ============= Part 3: Visualizing J(theta_0, theta_1) =============
print('\n===================Visualizing J(theta_0, theta_1) ...===================\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([theta0_vals.size, theta1_vals.size])

# Fill out J_vals
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
        t = np.row_stack((theta0_vals[i], theta1_vals[j]))
        J_vals[i, j] = computeCost(X, y, t)

# Surface plot
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
surffig = plt.figure(figsize=(10, 8), dpi=70)
ax3d = Axes3D(surffig)
ax3d.plot_surface(theta0_vals, theta1_vals, J_vals.T, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.title('Cost function')
plt.xlabel('theta0')
plt.ylabel('theta1')
m = cm.ScalarMappable(cmap=plt.get_cmap('rainbow'))
m.set_array(J_vals.T)
plt.colorbar(m)
plt.show()

# Contour plot
contourfig = plt.figure(figsize=(10, 8), dpi=70)
colorbar = plt.contourf(theta0_vals, theta1_vals, J_vals.T)
plt.contour(theta0_vals, theta1_vals, J_vals.T, colors='black', linewidths=1, linestyles='solid')
plt.plot(theta[0], theta[1], 'rx', linewidth=2)
plt.colorbar(colorbar)
plt.title('Cost function')
plt.xlabel('theta0')
plt.ylabel('theta1')
plt.ioff()
plt.show()
plt.close('all')




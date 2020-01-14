"""
@Author: Zijian Huang
@Contact: zijian_huang@sjtu.edu.cn
@File: ex1.py
@Time: 2020/1/08 8:21 PM

Machine Learning Online Class - Exercise 2: Logistic Regression
Instructions
------------
This file contains code that helps you get started on the logistic regression exercise.
Following functions are implemented:
    sigmoid.py
    costFunction.py
    gradientDescent.py
    featureNormalize.py
    plotData.py
    plotDecisionBoundary.py

Data:
    the first column of x refers to the score of exam 1
    the second column of x refers to the score of exam 2
    y refers to the admission decision [0, 1]
"""
import numpy as np

import matplotlib.pyplot as plt
import scipy.optimize as optim
from sigmoid import sigmoid
from featureNormalize import featureNormalize
from plotData import plotData
from plotDecisionBoundary import plotDecisionBoundary
from costFunction import costFunction
from gradientDescent import gradientDescent


# ==================== Part 1: Plotting ====================
# Load Data
data = np.loadtxt('./ex2data1.txt', dtype=float, delimiter=',')
X, y = data[:, :2], data[:, 2]
np.set_printoptions(precision=4, suppress=True)

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
fig = plotData(X, y, legend=['Admitted', 'Not admitted'], label=['Exam 1 score', 'Exam 2 score'])
plt.show()
plt.close(fig)

input("Program paused. Press enter to continue.\n")

# ============ Part 2: Compute Cost and Gradient ============
# Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
X = np.concatenate((np.ones([m, 1], dtype=float), X), axis=1)
initial_theta = np.zeros([n + 1])

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y, requires_grad=True)
print('Cost at initial theta (zeros): %f' % cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): \n', grad)
print('Expected gradients (approx):\n -0.1000, -12.0092, -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2])
cost, grad = costFunction(test_theta, X, y, requires_grad=True)
print('Cost at test theta: %f' % cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n', grad)
print('Expected gradients (approx):\n 0.043, 2.566, 2.647\n')

input("Program paused. Press enter to continue.\n")

# ============= Part 3: Optimizing using scipy.optimize.minimize  =============
res = optim.minimize(costFunction, initial_theta, (X, y), options={'maxiter': 400, 'disp': True})
theta = res.x
cost = res.fun

print('\nCost at theta found by optim: %f\n' % cost)
print('Expected cost (approx): 0.203\n')
print('theta: \n', theta)
print('Expected theta (approx):\n-25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta, X, y, legend=['Admitted', 'Not admitted'], label=['Exam 1 score', 'Exam 2 score'])
input("Program paused. Press enter to continue.\n")

# ============== Part 4: Predict and Accuracies ==============
# Predict probability for a student with score 45 on exam 1 and score 85 on exam 2
X_test = np.array([[1, 45, 85]])
prob = sigmoid(X_test @ theta)
print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = sigmoid(X @ theta)
pre = np.where(p > 0.5, 1, 0)
accuracy = np.mean((pre == y).astype(np.float)) * 100
print('Train Accuracy: %f' % accuracy)
print('Expected accuracy (approx): 89.0\n')

# ============= Part 5: Optimizing using gradientDescent  =============
# take feature normalize to accelerate the convergence
X, y = data[:, :2], data[:, 2]
X, mu, sigma = featureNormalize(X)
X = np.concatenate((np.ones([m, 1], dtype=float), X), axis=1)

iterations = 400
alpha = 2
theta, J_history = gradientDescent(X, y, initial_theta, alpha, iterations)
print('theta came up by gradient descent:\n', theta)

# plot convergence figure
plt.figure()
plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.title('Convergence graph of cost J')
plt.show()

# predict and accuracy
X_test = np.array([[45, 85]])
X_test = (X_test - mu) / sigma
prob = sigmoid(np.concatenate((np.array([[1.0]]), X_test), axis=1)  @ theta)
print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = sigmoid(X @ theta)
pre = np.where(p > 0.5, 1, 0)
accuracy = np.mean((pre == y).astype(np.float)) * 100
print('Train Accuracy: %f' % accuracy)
print('Expected accuracy (approx): 89.0\n')




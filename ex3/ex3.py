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


# =========== Part 1: Loading and Visualizing Data =============
print('Loading and Visualizing Data ...\n')
data = loadmat('ex3data1.mat')
X, y = data['X'], data['y']
displayData(X, image_size=(20, 20), disp_rows=10, disp_cols=10, padding=1, shuffle=True)

input("Program paused. Press enter to continue.\n")

# ============ Part 2a: Vectorize Logistic Regression ============


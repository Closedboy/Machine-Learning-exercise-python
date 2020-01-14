# -*- coding: utf-8 -*-

"""
@Author  : Zijian Huang
@Contact : zijian_huang@sjtu.edu.cn
@FileName: displayData.py
@Time    : 20-1-14 上午11:43
"""
import numpy as np
import matplotlib.pyplot as plt


def displayData(X, image_size, disp_rows=10, disp_cols=10, padding=1, shuffle=True):
    h, w = image_size
    if shuffle:
        np.random.shuffle(X)
    X = X[:disp_rows * disp_cols]
    disp_array = -np.ones([padding + disp_rows * (h + padding), padding + disp_cols * (w + padding)])
    for i in range(disp_cols):
        for j in range(disp_rows):
            pad_hidx = padding + j * (h + padding)
            pad_widx = padding + i * (w + padding)
            disp_array[pad_hidx: h + pad_hidx, pad_widx: w + pad_widx] = X[j + i * disp_rows].reshape(h, w)
    fig = plt.figure()
    plt.imshow(disp_array, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.close(fig)

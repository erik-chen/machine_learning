"""
    机器学习———罗杰斯回归
    作者：陈志豪
    版本：2.0
    日期：17/07/2019
    1.0：给定数据，求得Hypothesis
    2.0: 使用minimize方法，正则化
"""
import pandas as pd
import numpy as np
import scipy.optimize as op
import copy
from matplotlib import pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename, header=None)
    data = np.array(data)
    return data


def reshape_data(data):
    m, n = data.shape
    X = data[:, 0:n-1]
    y = data[:, n-1:n]
    return X, y


def init_data(X):
    m, n = X.shape
    initial_theta = np.zeros(n+1)
    X = np.hstack((np.ones((m, 1)), X))
    return X, initial_theta


def feature_scaling(X):
    n = X.shape[1]
    mean_X = []
    std_X = []
    for i in range(1, n):
        mean1 = np.mean(X[:, i])
        std1 = np.std(X[:, i])
        X[:, i] = (X[:, i] - mean1) / std1
        mean_X.append(mean1)
        std_X.append(std1)
    return X, mean_X, std_X


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def cost_function(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_reg = copy.deepcopy(theta)
    theta_reg[0, 0] = 0
    J = -1/m*(y.T@np.log(sigmoid(X@theta))+(1-y).T@np.log(sigmoid(-X@theta)))+lam/(2*m)*theta_reg.T@theta_reg
    return J


def gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_reg = copy.deepcopy(theta)
    theta_reg[0, 0] = 0
    grad = 1/m*X.T@(sigmoid(X@theta)-y)+lam/m*theta_reg
    return grad.flatten()


def adjust_theta(theta, mean_X, std_X):
    n = theta.shape[0]
    for i in range(n-1):
        theta[0] -= theta[i+1] * mean_X[i] / std_X[i]
    for i in range(n - 1):
        theta[i+1] = theta[i+1] / std_X[i]
    return theta


if __name__ == '__main__':
    data = load_data('D:\ex2data1.txt')
    X, y = reshape_data(data)
    X, initial_theta = init_data(X)
    lam = 0
    # X, mean_X, std_X = feature_scaling(X)
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='BFGS', jac=gradient)
    # result.x = adjust_theta(result.x, mean_X, std_X)
    print(result.x)
    theta = result.x.reshape((X.shape[1], 1))
    x0, y0 = [], []
    for i in range(X.shape[0]):
        x1=X[i:i+1, :]@theta
        x1=x1[0, 0]
        y1=-(y[i, 0]*np.log(sigmoid(x1))+(1-y[i, 0])*np.log(sigmoid(-x1)))
        x0.append(x1)
        y0.append(y1)

    print(x0, y0)
    ax1 = plt.subplot(111)
    ax1.scatter(x0, y0, s=30, c='r', marker='+')

    a, b, c, d = 0, 0, 0, 0
    a0, b0, c0, d0 = 0, 0, 0, 0
    for i in range(X.shape[0]):
        if x0[i]<0 and y0[i]<0.5:
            c += 1
            c0 += y0[i]
        elif x0[i]>0 and y0[i]<0.5:
            d += 1
            d0 += y0[i]
        elif x0[i]>0 and y0[i]>0.5:
            a += 1
            a0 += y0[i]
        elif x0[i]<0 and y0[i]>0.5:
            b += 1
            b0 += y0[i]
    print('a=', a, '和=', a0)
    print('b=', b, '和=', b0)
    print('c=', c, '和=', c0)
    print('d=', d, '和=', d0)
    plt.show()




"""
    机器学习———线性回归
    作者：陈志豪
    版本：5.0
    日期：16/07/2019
    1.0：给定数据，求得Hypothesis
    2.0：模块化代码
    3.0：适用于多变量线性回归
    4.0: 使用minimize方法
    5.0: 正则化
"""
import pandas as pd
import numpy as np
import scipy.optimize as op
import copy


def load_data(filename):
    data = pd.read_excel(filename)
    data = np.array(data)
    return data


def reshape_data(data):
    m, n = data.shape
    X = data[:, 0:n-1]
    print(X)
    y = data[:, n-1:n]
    return X, y


def init_data(X):
    m, n = X.shape
    initial_theta = np.zeros(n+1)
    X = np.hstack((np.ones((m, 1)), X))
    print(X)
    return X, initial_theta


def normal_equation(X, y):
    n = X.shape[1]
    reg_item = np.identity(n)
    reg_item[0, 0] = 0
    theta_normal_equation = np.linalg.pinv(X.T@X-lam*reg_item)@X.T@y
    return theta_normal_equation


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
    print(X)
    return X, mean_X, std_X


def cost_function(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_reg = copy.deepcopy(theta)
    theta_reg[0, 0] = 0
    J = 1/(2*m)*(X@theta-y).T@(X@theta-y)+lam/(2*m)*theta_reg.T@theta_reg
    print('J=', J)
    return J


def gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    theta_reg = copy.deepcopy(theta)
    theta_reg[0, 0] = 0
    grad = 1/m*X.T@(X@theta-y)+lam/m*theta_reg
    return grad.flatten()


def adjust_theta(theta, mean_X, std_X):
    n = theta.shape[0]
    for i in range(n-1):
        theta[0] -= theta[i+1] * mean_X[i] / std_X[i]
    for i in range(n - 1):
        theta[i+1] = theta[i+1] / std_X[i]
    return theta


if __name__ == '__main__':
    data = load_data('D:\ex1data2.xlsx')
    X, y = reshape_data(data)
    X, initial_theta = init_data(X)
    lam = 10
    theta_normal_equation = normal_equation(X, y)
    X, mean_X, std_X = feature_scaling(X)
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
    print(result)
    result.x = adjust_theta(result.x, mean_X, std_X)
    print(theta_normal_equation)
    theta = result.x.reshape(3, 1)
    print(theta)




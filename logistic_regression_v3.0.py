"""
    机器学习———罗杰斯回归
    作者：陈志豪
    版本：3.0
    日期：2019/8/20
    1.0：给定数据，求得Hypothesis
    2.0: 使用minimize方法，正则化
    3.0: 应用于24点
"""
import pandas as pd
import numpy as np
import scipy.optimize as op


def load_data(filename):
    data_pandas = pd.read_excel(filename)
    data = np.array(data_pandas)
    return data


def get_X_y(data):
    X, y = data[:, 0:4], data[:, 4:]
    X0 = np.ones((715, 1))
    X = np.hstack((X0, X))
    return X, y


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def cost_function(theta, X, y):
    theta = np.reshape(theta, (5, 1))
    J = -1/715*(y.T@np.log(sigmoid(X@theta))/np.log(np.e)+(1-y).T@np.log(sigmoid(-X@theta))/np.log(np.e))
    return J


def gradient(theta, X, y):
    theta = np.reshape(theta, (5, 1))
    grad = 1/715*X.T@(sigmoid(X@theta)-y)
    return grad.flatten()


def main():
    data = load_data('D:\eqwc3.xlsx')
    X, y = get_X_y(data)
    initial_theta = np.array([1, 2, 3, 4, 0])
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
    print(result)


if __name__ == '__main__':
    main()
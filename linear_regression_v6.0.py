"""
    机器学习———线性回归
    作者：陈志豪
    版本：6.0
    日期：2019/08/15
    1.0：给定数据，求得Hypothesis
    2.0：模块化代码
    3.0：适用于多变量线性回归
    4.0: 使用minimize方法
    5.0: 正则化
    6.0: 重写代码
"""
import pandas as pd
import numpy as np
import scipy.optimize as op


def load_data(filename):
    data_pandas = pd.read_excel(filename)
    data = np.array(data_pandas)  # data = data_pandas.values
    return data


def get_X_y(data):
    X, y = data[:, 0:2], data[:, 2:]
    X0 = np.ones((47, 1))
    X = np.hstack((X0, X))
    return X, y


def feature_scaling(X):
    X_mean_list = []
    X_std_list = []
    for i in range(1, 3):
        mean = np.mean(X[:, i])
        std = np.std(X[:, i])
        X[:, i] = (X[:, i] - mean) / std
        X_mean_list.append(mean)
        X_std_list.append(std)
    return X, X_mean_list, X_std_list


def cost_function(theta, X, y):
    theta = np.reshape(theta, (3, 1))  # theta = theta.reshape((2, 1))
    J = 1/(2*47) * (X @ theta - y).T @ (X @ theta - y)
    print(J)
    return J


def gradient(theta, X, y):
    theta = np.reshape(theta, (3, 1))
    grad = 1/47 * X.T @ (X @ theta - y)
    return grad.flatten()


def main():
    data = load_data('ex1data2.xlsx')
    X, y = get_X_y(data)
    X, X_mean_list, X_std_list = feature_scaling(X)
    initial_theta = np.ones(3)
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y), method='TNC', jac=gradient)
    print(result)


if __name__ == '__main__':
    main()


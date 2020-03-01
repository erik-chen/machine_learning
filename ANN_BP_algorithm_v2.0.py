"""
    机器学习———神经网络BP算法
    作者：陈志豪
    版本：1.0
    日期：2019/8/20
    1.0：
    2.0
"""
import pandas as pd
import numpy as np
import scipy.optimize as op
import copy

def load_data(filename):
    data_pandas = pd.read_excel(filename)
    data = np.array(data_pandas)
    return data


def get_X_y(data):
    X, y = data[:, 0:2], data[:, 2:]
    X0 = np.ones((400, 1))
    X = np.hstack((X0, X))
    return X, y


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def cost_function(Theta, X, y):
    Theta_1, Theta_2 = np.split(Theta, (6,))[0], np.split(Theta, (6,))[1]
    Theta_1 = np.reshape(Theta_1, (2, 3))
    Theta_2 = np.reshape(Theta_2, (1, 3))
    J = 0
    for i in range(400):
        a_1 = X[i, :]
        a_1 = np.reshape(a_1, (3, 1))
        a_2 = sigmoid(Theta_1@a_1)
        a_2 = np.vstack((np.array([[1]]), a_2))
        a_3 = sigmoid(Theta_2@a_2)
        y_i = y[i:i+1, :]
        J += y_i.T@np.log(a_3)+(1-y_i).T@np.log(1-a_3)
    J /= -400

    return J


def gradient(Theta, X, y):
    Theta_1, Theta_2 = np.split(Theta, (6,))[0], np.split(Theta, (6,))[1]
    Theta_1 = np.reshape(Theta_1, (2, 3))
    Theta_2 = np.reshape(Theta_2, (1, 3))
    Delta_2 = np.ones((1, 3))
    Delta_1 = np.ones((2, 3))
    for i in range(400):
        a_1 = X[i, :]
        a_1 = np.reshape(a_1, (3, 1))
        a_2 = sigmoid(Theta_1 @ a_1)
        a_2 = np.vstack((np.array([[1]]), a_2))
        a_3 = sigmoid(Theta_2 @ a_2)
        y_i = y[i:i + 1, :]
        delta_3 =  a_3 - y_i  # (1, 1)
        delta_2 = Theta_2.T @ delta_3 * a_2 * (1 - a_2)  # (5, 1) @ (1, 1) = (5, 1)
        delta_2 = np.vsplit(delta_2, (1,))[1]
        Delta_2 += delta_3 @ a_2.T
        Delta_1 += delta_2 @ a_1.T
    D_2 = Delta_2 / 400
    D_1 = Delta_1 / 400
    D_2 = D_2.flatten()
    D_1 = D_1.flatten()
    grad = np.hstack((D_1, D_2))

    print(grad)
    return grad


def main():
    data = load_data('D:\weqqw.xlsx')
    X, y = get_X_y(data)
    initial_Theta = np.ones(9)
    # epsilon = 1e-4
    # for i in range(9):
    #     Theta_big = copy.deepcopy(initial_Theta)
    #     Theta_big[i] += epsilon
    #     Theta_small = copy.deepcopy(initial_Theta)
    #     Theta_small[i] -= epsilon
    #     print((cost_function(Theta_big, X, y)-cost_function(Theta_small, X, y))/2/epsilon)

    result = op.minimize(fun=cost_function, x0=initial_Theta, args=(X, y), method='TNC', jac=gradient, options={'maxiter': 1000000})
    print(result)


if __name__ == '__main__':
    main()
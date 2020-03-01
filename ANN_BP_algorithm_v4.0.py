"""
    机器学习———神经网络BP算法
    作者：陈志豪
    版本：4.0
    日期：2019/8/22

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
    X, y = data[:, 0:4], data[:, 4:]
    X0 = np.ones((715, 1))
    X = np.hstack((X0, X))
    return X, y


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def cost_func(theta, X, y):
    Theta1 = np.reshape(theta[:25], (5, 5))
    Theta2 = np.reshape(theta[25:], (1, 6))
    # print('Theta1=', Theta1)
    # print('Theta2=', Theta2)
    J = 0
    for i in range(715):
        a1 = np.reshape(X[i], (5, 1))
        # print(a1)
        a2 = sigmoid(Theta1 @ a1)
        a2 = np.vstack((np.ones((1, 1)), a2))
        # print(a2)
        a3 = sigmoid(Theta2 @ a2)
        a4 = sigmoid(-Theta2 @ a2)
        yi = y[i:i+1]
        # J += yi * np.log(a3) + (1 - yi) * np.log(1 - a3)
        J += yi * np.log(a3) + (1 - yi) * np.log(a4)
    J /= -715
    print('J=', J)
    return J


def gradient(theta, X, y):
    print('theta=', theta)
    Theta1 = np.reshape(theta[:25], (5, 5))
    Theta2 = np.reshape(theta[25:], (1, 6))
    Delta2 = np.zeros((1, 6))
    Delta1 = np.zeros((5, 5))
    # print(Delta2)
    # print(Delta1)
    for i in range(715):
        a1 = np.reshape(X[i], (5, 1))
        # print(a1)
        a2 = sigmoid(Theta1 @ a1)
        a2 = np.vstack((np.ones((1, 1)), a2))
        # print(a2)
        a3 = sigmoid(Theta2 @ a2)
        # print(a3)
        yi = y[i:i+1]
        # print(yi)
        delta3 = a3 - yi
        # print(delta3)
        delta2 = Theta2.T @ delta3 * a2 * (1 - a2)
        delta2 = np.vsplit(delta2, (1,))[1]
        # print(delta2)
        # delta1 = Theta1.T @ delta2 * a1 * (1 - a1)
        # delta1 = np.vsplit(delta1, (1,))[1]
        # # print(delta1)
        Delta2 += delta3 @ a2.T
        Delta1 += delta2 @ a1.T
    D1 = Delta1 / 715
    D2 = Delta2 / 715
    # print('D2=', D2)
    D = np.hstack((D1.flatten(), D2.flatten()))
    print('D=', D)
    return D


def main():
    data = load_data('D:\eqwc3.xlsx')
    X, y = get_X_y(data)
    # print('X=', X)
    # print('y=', y)
    initial_theta = np.random.uniform(-0.02, 0.02, 31)
    print('initial_theta=', initial_theta)
    # initial_theta = [-0.0015514, - 0.00504088, 0.01258308, - 0.00823796, - 0.01703769, 0.00295712,
    #  - 0.01912241, 0.01611616, - 0.01403094, - 0.01538667, 0.01526665, - 0.01099241,
    #  - 0.0119332]
    # initial_theta = np.array([-0.00657367, 0.00818979, 0.00093083, 0.00401997, 0.01458579, 0.00591292,
    #                           -0.0048265, -0.0090128, -0.0001452, -0.00553074, -0.0086425, 0.00391508,
    #                           0.01190323])
    # print(initial_theta)
    # cost_func(initial_theta, X, y)
    gradient(initial_theta, X, y)
    """
        check gradient
    """
    epsilon = 1e-4
    D_check = np.zeros(31)
    for i in range(31):
        theta_big = copy.deepcopy(initial_theta)
        theta_big[i] += epsilon
        theta_small = copy.deepcopy(initial_theta)
        theta_small[i] -= epsilon
        D_check[i] = (cost_func(theta_big, X, y) - cost_func(theta_small, X, y))/2/epsilon
    print('D_check=', D_check)
    result = op.minimize(fun=cost_func, x0=initial_theta, args=(X, y), method='TNC', jac=gradient,
                         options={'maxiter': 1000000})
    print(result)




if __name__ == '__main__':
    main()
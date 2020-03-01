"""
    机器学习———罗杰斯回归
    作者：陈志豪
    版本：1.0
    日期：01/07/2019
    1.0：给定数据，求得Hypothesis
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy


def data_leading_in():
    """
        导入txt数据，return矩阵
    """
    # 用pandas导入excel数据，导入的数据类型为pandas.core.frame.DataFrame
    data_pandas = pd.read_excel(r'C:\Users\Administrator\Desktop\logistic_data.xlsx', header=None)
    # 把数据类型转化为numpy.ndarray，即矩阵
    data = data_pandas.values
    return data


def set_parameter():
    """
        设置参数：1、迭代次数 2、学习速率 3、初始回归系数
    """
    iterations = 1000
    alpha = 0.003
    theta = np.array([[-25.16133356],
                      [0.20623171],
                      [0.2014716]])
    return iterations, alpha, theta


def draw_data():
    """
        画散点图
    """
    x1_pos = []
    x2_pos = []
    x1_neg = []
    x2_neg = []
    for i in range(m):
        x1_pos.append(data[i, 0]) if data[i, 2] else x1_neg.append(data[i, 0])
        x2_pos.append(data[i, 1]) if data[i, 2] else x2_neg.append(data[i, 1])
    ax1 = plt.subplot(211)
    ax1.scatter(x1_pos, x2_pos, s=30, c='r', marker='+')
    ax1.scatter(x1_neg, x2_neg, s=30, c=[[0, 1, 0]], marker='o')
    ax1.set_xlabel('Exam 1 score')
    ax1.set_ylabel('Exam 2 score')


def g(x):
    a = 1 / (1+np.exp(-x))
    return a


# def draw_f():
#     x1= np.linspace(-10, 10, 2001)
#     y1 = g(x1)
#     ax1 = plt.subplot(312)
#     # 画出图像
#     ax1.plot(x1, y1)


def feature_scaling():
    """
        特征缩放&均值归一化，使得-1≤X[1]≤1
    """
    mean_X_list = []
    std_X_list = []
    for i in range(n - 1):
        mean_X = np.mean(X_without_x0[:, i])
        std_X = np.std(X_without_x0[:, i])
        X_without_x0[:, i] = (X_without_x0[:, i] - mean_X) / std_X * 2
        mean_X_list.append(mean_X)
        std_X_list.append(std_X)
    return X_without_x0, mean_X_list, std_X_list


def gradient_descent():
    """
        梯度下降
    """
    global theta
    # (x1,y1)构成代价函数值关于迭代次数的图像
    x1 = []
    y1 = []
    for i in range(iterations):
        # i累加到列表x1中
        x1.append(i + 1)
        # 迭代theta
        delta = 1 / m * X.T @ (g(X @ theta) - y)
        theta = theta - alpha * delta
        # 迭代计算代价函数J的值，累加到列表y1中
        J = -1/m*(y.T@np.log(g(X @ theta))+(1-y).T@np.log(g(-X @ theta)))
        y1 += J.tolist()
        # print(theta, J)
    return x1, y1


def adjust_theta():
    """
        由于特征缩放&均值归一化会扭曲hypothesis图像，所以要调整回来
    """
    for i in range(n - 1):
        theta[0] -= 2*theta[i+1] * mean_X_list[i] / std_X_list[i]
    for i in range(n - 1):
        theta[i+1] = 2 * theta[i+1] / std_X_list[i]
    # print(theta)


def draw_gradient_descent():
    """
        绘制代价函数值关于迭代次数的图像
    """
    # 设置上半张画布
    ax2 = plt.subplot(212)
    # 画出图像
    ax2.plot(x1, y1)
    ax2.set_title('Gradient Descent')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cost Function Value')


if __name__ == '__main__':
    data = data_leading_in()
    print(data, data.shape)
    # m是矩阵的行数,n是矩阵的列数
    m, n = data.shape
    draw_data()
    # 设置参数
    iterations, alpha, theta = set_parameter()
    # 从data中得到X_without_x0和y
    X_without_x0 = copy.deepcopy(np.hsplit(data, (n - 1, n))[0])
    X_without_x0 = X_without_x0.astype(float)
    y = np.hsplit(data, n)[n - 1]
    # # 正规方程
    # normal_equation()
    # 特征缩放和均值归一化
    X_without_x0, mean_X_list, std_X_list = feature_scaling()
    # X_without_x0加上x0项得到X
    X = np.hstack((np.ones((m, 1)), X_without_x0))
    # 梯度下降
    x1, y1 = gradient_descent()
    # # 绘制梯度下降图像，观察迭代次数
    draw_gradient_descent()
    # 调整theta，消除特征缩放对图像的影响
    adjust_theta()
    plt.show()





"""
    机器学习———罗杰斯回归
    作者：陈志豪
    版本：2.0
    日期：17/07/2019
    2.0: 观察图像
"""
import pandas as pd
import numpy as np
import scipy.optimize as op
import copy
from matplotlib import pyplot as plt


def load_data(filename):
    data = pd.read_excel(filename, header=None)
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
    data = load_data(r'C:\Users\Administrator\Desktop\logistic_data.xlsx')
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

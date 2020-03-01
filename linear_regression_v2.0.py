"""
    机器学习———线性回归
    作者：陈志豪
    版本：2.0
    日期：28/06/2019
    1.0：给定数据，求得Hypothesis
    2.0：模块化代码
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import copy


def data_leading_in():
    """
        导入excel数据，return矩阵
    """
    # 用pandas导入excel数据，导入的数据类型为pandas.core.frame.DataFrame
    data_pandas = pd.read_excel('D:\ex1data1.xlsx')
    # 把数据类型转化为numpy.ndarray，即矩阵
    data = data_pandas.values
    return data


def set_parameter():
    """
        设置参数：1、迭代次数 2、学习速率 3、初始回归系数
    """
    iterations = 3000
    alpha = 0.01
    theta = np.ones((n, 1))
    return iterations, alpha, theta


def normal_equation():
    """
        正规方程
    """
    X = np.hstack((np.ones((m, 1)), X_without_x0))
    y = np.hsplit(data, n)[n - 1]
    theta_normal_equation = np.linalg.pinv(X.T@X)@X.T@y
    print(theta_normal_equation)


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


def adjust_theta():
    """
        由于特征缩放&均值归一化会扭曲hypothesis图像，所以要调整回来
    """
    for i in range(n - 1):
        theta[0] -= 2*theta[i+1] * mean_X_list[i] / std_X_list[i]
    for i in range(n - 1):
        theta[i+1] = 2 * theta[i+1] / std_X_list[i]
    print(theta)
    return theta


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
        delta = 1 / m * X.T @ (X @ theta - y)
        theta = theta - alpha * delta
        # 迭代计算代价函数J的值，累加到列表y1中
        sqrError = (X @ theta - y) * (X @ theta - y)
        J = 1 / (2 * m) * sum(sqrError)
        y1 += J.tolist()
    return x1, y1


def draw_gradient_descent():
    """
        绘制代价函数值关于迭代次数的图像
    """
    # 设置上半张画布
    ax1 = plt.subplot(111)
    # 画出图像
    ax1.plot(x1, y1)
    ax1.set_title('Gradient Descent')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cost Fuction Value')


def draw_hypothesis():
    """
        绘制目标函数hypothesis的图像
    """
    # (x2,y2)构成目标函数hypothesis的图像
    x2 = list(range(5, 25))
    y2 = list(map(lambda x: theta[0]+theta[1]*x, x2))
    # 设置下半张画布
    ax2 = plt.subplot(212)
    # 画出图像
    ax2.plot(x2, y2)
    ax2.set_title('Hypothesis')
    ax2.set_xlabel('Population')
    ax2.set_ylabel('Profits')
    # ax2 在后面还会用到
    return ax2


def draw_data():
    """
        绘制散点图
    """
    # (x3,y3)构成散点图
    x3 = np.hsplit(data, 2)[0]
    y3 = np.hsplit(data, 2)[1]
    # 画出图像
    ax2.scatter(x3, y3, s=30, c='r', marker='x')


if __name__ == '__main__':
    # 导入数据
    data = data_leading_in()
    # m是矩阵的行数,n是矩阵的列数
    m, n = data.shape
    # 设置参数
    iterations, alpha, theta = set_parameter()
    # 从data中得到X_without_x0和y
    X_without_x0 = copy.deepcopy(np.hsplit(data, (n - 1, n))[0])
    y = np.hsplit(data, n)[n - 1]
    # 正规方程
    normal_equation()
    # 特征缩放和均值归一化
    X_without_x0, mean_X_list, std_X_list = feature_scaling()
    # X_without_x0加上x0项得到X
    X = np.hstack((np.ones((m, 1)), X_without_x0))
    # 梯度下降
    x1, y1 = gradient_descent()
    # 绘制梯度下降图像，观察迭代次数
    draw_gradient_descent()
    # 调整theta，消除特征缩放对图像的影响
    # theta = adjust_theta()
    # # 绘制目标函数
    # ax2 = draw_hypothesis()
    # # 绘制散点图
    # draw_data()
    plt.show()


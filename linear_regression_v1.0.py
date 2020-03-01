def main():
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    # 数据导入
    ex1data1 = pd.read_excel('D:\ex1data1.xlsx')
    # 转为矩阵
    ex1data1 = ex1data1.values
    m = ex1data1.shape[0]
    mean_X = sum(np.hsplit(ex1data1, 2)[0])/m
    range_X = max(np.hsplit(ex1data1, 2)[0])-min(np.hsplit(ex1data1, 2)[0])
    X = np.hstack((np.ones((m, 1)), ((np.hsplit(ex1data1, 2)[0]-mean_X) / range_X)*2))
    y = np.hsplit(ex1data1, 2)[1]
    iterations = 6000
    alpha = 0.01
    theta = np.array([[0],
                      [0]])
    x1 = []
    y1 = []
    for i in range(iterations):

        # theta 迭代
        delta = 1 / m * X.T@(X@theta-y)
        theta = theta - alpha * delta
        x1.append(i+1)
        # 计算J
        sqrError = (X @ theta - y) * (X @ theta - y)
        J = 1 / (2 * m) * sum(sqrError)
        y1 += J.tolist()
    ax1 = plt.subplot(211)
    ax1.plot(x1, y1)
    x2 = np.arange(5, 25)
    theta[0] = theta[0] - 2*theta[1]*mean_X/range_X
    theta[1] = 2*theta[1]/range_X
    y2 = theta[0]+theta[1]*x2
    x3 = np.hsplit(ex1data1, 2)[0]
    y3 = np.hsplit(ex1data1, 2)[1]
    ax2 = plt.subplot(212)
    ax2.plot(x2, y2)
    ax2.scatter(x3, y3, s=30, c='r', marker='x')
    plt.show()


if __name__ == '__main__':
    main()

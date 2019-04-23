'''
Logistic Regression Working
@auther bubble captain
'''
from numpy import *
import  numpy as np
from matplotlib import pyplot as pl


'''把数据分别读取到数据集和特征标签集里'''
def readDataSet(file_n):
    dataMat = [];  # 数据集
    labelMat = []  # 标签集
    filepoint = open(file_n)
    for line in filepoint.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 把数据特征提取出来写进数据集矩阵里面
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat
"""
sigmoid激活函数
"""
def sigmoid(result):
    return 1.0 / (1 + exp(-result))
"""
梯度下降求解回归函数a,控制条件是迭代次数,无正则项
"""

'''随意梯度下降法优化'''
def stoc_grad_ascent_one(dataMatIn, classLabels):
    count = 150
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    for j in range(count):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1 + i + j) + 0.01 #保证多次迭代后新数据仍然有影响力
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[i] * weights))  # 数值计算
            error = classLabels[i] - h
            a=[0,0,0]
            for j in range(0,n):
                a[j]=  dataMatIn[i][j]*alpha * error
            weights = weights + a
            del(dataIndex[randIndex])
    return weights

"""
将数据展示出来
"""
def plotBestFit(weights, dataMat,labelMat):
    dataArr = array(dataMat)
    weights = weights.transpose()
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    print(xcord1)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = ((-weights[0]-weights[1]*x)/weights[2])
    ax.plot(x, y)
    pl.title('SGD no reg')
    pl.xlabel('X1')
    pl.ylabel('Y1')
    pl.show()

"""
测试函数
"""
def logistic_regression():

    filename="simpleTrainset.txt"  #高斯分布数据
    dataMat,labelMat = readDataSet(filename)    #读入本地数据文件
    A= stoc_grad_ascent_one(dataMat,labelMat)   #回归系数a的值
    plotBestFit(A, dataMat, labelMat)

logistic_regression()

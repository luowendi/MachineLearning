'''
Logistic Regression Working
@auther bubble captain
'''
from numpy import *
import  numpy as np
from matplotlib import pyplot as pl


'''把数据分别读取到数据集和特征标签集里'''
def readDataSet(file_n):
    dataMat = []  # 数据集
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
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    learn_rate = 0.001     #设置学习率
    count = 1000       #设置迭代次数
    weights = np.ones((n,1))     #全部初始化为1
    for k in range(count):
        t = sigmoid(dataMatrix*weights)       #激活函数预测
        error=(labelMat-t)                    #真实值与预测值的误差计算
        temp = dataMatrix.transpose()*error   #交叉熵cost函数对所有参数的偏导数
        weights=weights+learn_rate*(temp) #不断更新处于对数指数位置的权重
    return array(weights.transpose())              #返回已经训练好的（W 特征）


"""
将数据展示出来
"""
def plotBestFit(weights, dataMat,labelMat):
    dataArr = array(dataMat)
    weights = weights.transpose()
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    pl.xlabel('X1')
    pl.ylabel('Y1')
    pl.show()

def plotBestFit1(weights, dataMat,labelMat):
    dataArr = array(dataMat)
    weights = weights.transpose()
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

    fig = pl.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3,3, 1)
    y = ((weights[0]+weights[1]*x)/weights[2])
    ax.plot(x, y)
    pl.title('Grad no reg')
    pl.xlabel('X1')
    pl.ylabel('Y1')
    pl.show()
"""
测试函数
"""
def logistic_regression():
   # filename=input("请输入数据文件名称:")

    filename="simpleTrainset.txt"  #高斯分布数据

    #filename = "simpleTrainset.txt"    #马死亡数据
    dataMat,labelMat = readDataSet(filename)    #读入本地数据文件
    A= gradAscent(dataMat,labelMat)   #回归系数a的值
    plotBestFit(A, dataMat, labelMat)
    #plotBestFit1(A, dataMat, labelMat)

logistic_regression()

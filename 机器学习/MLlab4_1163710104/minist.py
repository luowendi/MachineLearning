import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import tensorflow.examples.tutorials.mnist.input_data as mnist_input_data
from PIL import Image
import ssl


import input_data

 #取消签名验证
ssl._create_default_https_context = ssl._create_unverified_context

def mean_to_zero(dataMat):
    data_mean = np.mean(dataMat, axis=0) # 按照列求平均值，即求各个纬度的平均值
    data_zero_mean = dataMat - data_mean
    return data_zero_mean, data_mean

def change_data(data, eigVals, eigVects):

    def decide_k(eigVals, m, percent = 0.99):
        sum1 = sum(eigVals)
        eigVals_sort = np.sort(eigVals)
        eigVals_sort = eigVals_sort[::-1]
        sum2 = 0
        for i in range(m):
            sum2 += eigVals_sort[i]
            if sum2 / sum1 > percent:
                break
        return (i + 1)
    k = decide_k(eigVals, data.shape[1])
    eigValIndice = np.argsort(eigVals) #对特征值升序排列
    n_eigValIndice = eigValIndice[-1:-(k + 1):-1] # 最大k个特征值的下标
    U = []
    for i in range(k - 1):
        if i == 0:
            U = np.vstack((eigVects[n_eigValIndice[i]], eigVects[n_eigValIndice[i + 1]]))
        else:
            U = np.vstack((U, eigVects[n_eigValIndice[i + 1]]))
    data_rot = np.array(U).dot(data.transpose())
    return data_rot.transpose(), U


def restore_data(data_rot, U, data_mean):
    restored_data = U.transpose().dot(data_rot.transpose()) + data_mean.reshape(-1, 1)
    return np.array(restored_data)


def PCA(dataMat):

    data, data_mean = mean_to_zero(dataMat) #将数据的均值归一化到零

    m = data.shape[1]
    cov = np.zeros((m, m))
    for i in range(len(data)):
        cov += data[i].reshape(1, -1).transpose().dot(data[i].reshape(1, -1))
    cov = cov / m
    eigVals, eigVects = np.linalg.eig(np.mat(cov)) #求特征值和特征向量

    data_rot, U = change_data(data, eigVals, eigVects) # 旋转、降维

    old_restored_data = U.transpose().dot(data_rot.transpose()) + data_mean.reshape(-1, 1)
    restored_data = np.array(old_restored_data) # 恢复数据
    return data_rot, restored_data, U

def mnist():

    #以下是根据网上教程的编写代码,mnist下载数据集
    mnist = mnist_input_data.read_data_sets("MNIST_data/", one_hot=False)

    imgs = mnist.train.images
    labels = mnist.train.labels

    origin_7_imgs = []
    for i in range(1000):
        if labels[i] == 7 and len(origin_7_imgs) < 100:
            origin_7_imgs.append(imgs[i])
    def array_to_image(array):
        array = array * 255
        new_img = Image.fromarray(array.astype(np.uint8))
        return new_img
    def comb_imgs(origin_imgs, col, row, each_width, each_height, new_type):
        new_img = Image.new(new_type, (col * each_width, row * each_height,))
        for i in range(len(origin_imgs)):
            each_img = array_to_image(np.array(origin_imgs[i]).reshape(each_width, each_width))
            new_img.paste(each_img, ((i % col) * each_width, (i // col) * each_width))
        return new_img

    hundred_origin_7_imgs = comb_imgs(origin_7_imgs, 10, 10, 28, 28, 'L')
    hundred_origin_7_imgs.show()

    low_d_feat_for_7_imgs, restored_imgs, U = PCA(np.array(origin_7_imgs))
    low_d_img = comb_imgs(restored_imgs.transpose(), 10, 10, 28, 28, 'L')
    low_d_img.show()


mnist()



import numpy as np
class PCA:
    def __init__(self, X):
        self.X = X
        self.mean = None
        self.feature = None
    def transform(self, n_components=1):
        n_samples, n_features = self.X.shape  # 数据的样本数和特征数
        self.mean = np.array([np.mean(self.X[:, i]) for i in range(n_features)])  # 计算每一个维度的均值
        # 去除平均值
        norm_X = self.X - self.mean
        # 计算散度矩阵
        scatter_matrix = np.dot(norm_X.T, norm_X)
        # 计算散度矩阵的特征向量和特征值
        eig_val, eig_vec = np.linalg.eig(scatter_matrix)
        # 根据特征值的大小，倒叙排序特征向量
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
        eig_pairs.sort(reverse=True)
        # 选取前n个特征向量
        self.feature = np.array([ele[1] for ele in eig_pairs[: n_components]])
        # 转化得到降维的数据
        new_data = np.dot(norm_X, self.feature.T)
        return new_data
    def inverse_transform(self, new_data):
        return np.dot(new_data, self.feature) + self.mean

def loadDataSet(fileName):
    return np.loadtxt(fileName, dtype=np.float)

if __name__ == "__main__":
    X = loadDataSet('./data/testPCA4.txt')
    myPCA = PCA(X=X)
    new_data = myPCA.transform(n_components=1)
    print("降维后数据")
    print(new_data)
    origin_data = myPCA.inverse_transform(new_data=new_data)
    print("降维前数据")
    print(origin_data)
    print('降维成{}维数据后，前后方差比为：{}'.format(new_data.shape[1], np.var(origin_data) / np.var(X)))

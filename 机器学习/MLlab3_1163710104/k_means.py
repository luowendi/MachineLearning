import numpy as np
import operator
import matplotlib.pyplot as plt
import matplotlib.cm as cm

r = lambda: np.random.randint(1, 100)

class Centroid:

    def __init__(self, pos):
        self.pos = pos
        self.points = []
        self.previous_points = []
        self.color = None

class KMeans:
    """
    无监督学习
    """
    def __init__(self, n_centroids=5):
        self.n_centroids = n_centroids
        self.centroids = []

        # generate initial centroids
        for _ in range(n_centroids):
            self.centroids.append(Centroid(np.array([r(), r()])))
        
        # assign a color to each centroid
        colors = cm.rainbow(np.linspace(0, 1, len(self.centroids)))
        for i, c in enumerate(self.centroids):
            c.color = colors[i]

    def sample_data(self, samples=50):
        """
       为类变量复值，随机生成数据，进行分类
        """
        self.X = [[r(), r()] for _ in range(samples)]

    def fit(self):
        """
        对生成的数据进行聚类分析
        """
        self.n_iters = 0
        fit = False 
        while not fit:
            for point in self.X:
                closest = self.assign_centroid(point)
                closest.points.append(point)

            # if length of array of centroids that did not change == number of centroids
            if len([c for c in self.centroids if c.points == c.previous_points]) == self.n_centroids:
                fit = True
                self._update_centroids(reset=False)
            else:
                self._update_centroids()

            self.n_iters += 1


    def assign_centroid(self, x):

        distances = {}
        for centroid in self.centroids:
            distances[centroid] = np.linalg.norm(centroid.pos - x)
        closest = min(distances.items(), key=operator.itemgetter(1))[0]
        return closest


    def _update_centroids(self, reset=True):

        for centroid in self.centroids:
            centroid.previous_points = centroid.points
            x_cor = [x[0] for x in centroid.points]
            y_cor = [y[1] for y in centroid.points]
            try:
                centroid.pos[0] = sum(x_cor)/len(x_cor)
                centroid.pos[1] = sum(y_cor)/len(y_cor)
            except:
                pass

            if reset:
                centroid.points = []
        
    def show(self):


        for i, c in enumerate(self.centroids):
            plt.scatter(c.pos[0], c.pos[1], marker='o', color=c.color, s=75)
            x_cors = [x[0] for x in c.points]
            y_cors = [y[1] for y in c.points]
            plt.scatter(x_cors, y_cors, marker='.', color=c.color)

        title = 'K-Means'
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.savefig('{}.png'.format(title))
        plt.show()



if __name__ == '__main__':
    kmeans = KMeans(n_centroids=4)
    kmeans.sample_data()
    kmeans.fit()
    kmeans.show()


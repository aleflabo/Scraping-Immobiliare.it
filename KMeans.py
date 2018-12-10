import numpy as np

class KMeans:

    def updateClusters(self):
        clus = {i : [] for i in range(self.k)}
        for i in range(self.m):
            dist = np.sum(np.array(self.X[i] - self.centroids)**2, axis=1)
            clus[np.argmin(dist)] += [i]
        self.clustering = clus

    def updateCentroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.X[self.clustering[i]], axis=0)

    def __init__(self, X, k, max_itr=1000):
        self.X, self.k = X, k
        self.m, self.d = X.shape
        self.centroids = X[np.random.choice(self.m, k, replace=False)]
        self.iter = 0
        while True:
            self.updateClusters()
            oldCentroids = self.centroids.copy()
            self.updateCentroids()
            self.iter += 1
            if (self.centroids == oldCentroids).all() or self.iter >= max_itr:
                break
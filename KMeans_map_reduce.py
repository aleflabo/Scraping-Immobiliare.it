import numpy as np
class KMeans_map_reduce:

    def updateClusters(self):
        clus = {i : [] for i in range(self.k)}
        daje = lambda x,y : j.append((y[1]))
        fun = (list(map(lambda x: ((np.argmin(np.sum(np.array(x - self.centroids)**2,axis=1))),x), self.X)))
        for i in range(self.k):
            try:
                j = [(list(filter(lambda x: x[0] == i , fun)))[0][1]]
                reduce( daje, list(filter(lambda x: x[0] == i, fun)))
            except: 
                j = [0]           
            clus[i] = j
        self.clustering = clus
        
    def updateCentroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.clustering[i], axis=0)

    def __init__(self, X, k, max_itr=10000):
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
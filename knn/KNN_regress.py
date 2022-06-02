import numpy as np
class KNN1:
    def __init__(self, n_neighbors, use_KDtree):
        self.n_neighbors = n_neighbors
        self.use_KDtree = use_KDtree

    def __euclidean_distance(self, x1, x2):
        euc_dist = np.sqrt(np.sum((x1 - x2) ** 2))
        return euc_dist

    def __get_neighbors(self, x):
        distances = []
        for x_train in self.X_train:
            distances.append(self.__euclidean_distance(x, x_train))
        distances_sort = distances.copy()
        distances_sort.sort()
        distances_sort = distances_sort[0:self.n_neighbors]
        neighbors = []
        for d in distances_sort:
            neighbors.append(distances.index(d))
        return neighbors

    def __regression(self, neighbors):
        val = []
        for n in neighbors:
            val.append(self.y_train[n])
        avg = np.average(val)
        return avg

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        results = []
        for x in X:
            nidx = self.__get_neighbors(x)
            results.append(self.__regression(nidx))
        return np.asarray(results)
    def score(self,x,y):
        sum = 0
        for i in range(len(x)):
            sum +=(x[i]-y[i])**2
        return sum/len(x)

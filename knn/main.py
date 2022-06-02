import numpy as np
from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))
class KNN:
    def __init__(self, n_neighbors=3,useKDTree=False):
        self.k = n_neighbors
        self.use = useKDTree

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):

        if(self.use == True):
            from sklearn.neighbors import KDTree
            tree = KDTree(x, leaf_size=2)
            dist, ind = tree.query(x[::], self.k)

            print(ind)
        elif(self.use == False):
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_idx = np.argsort(distances)[: self.k]
            k_neighbor_labels = [self.y_train[i] for i in k_idx]
            most_common = Counter(k_neighbor_labels).most_common(1)
            return most_common[0][0]
    def score(self,predi):
        predictions = predi
        acc = np.sum(predictions == self.y_train)/len(self.y_train)
        return acc

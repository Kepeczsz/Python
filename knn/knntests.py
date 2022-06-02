import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import datasets
from KNN_regress import  KNN1
from main import KNN
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import svm
from pcamoje import pca1
from sklearn.impute import SimpleImputer
X,y = datasets.make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    random_state=3
)

# zad1
clf1 = KNN(n_neighbors=5,useKDTree=False)
clf1.fit(X,y)
prediction = clf1.predict(X)
accuracy = clf1.score(prediction)
print(accuracy)


classification = clf1.predict(X)
print("kasyfikacja", classification)


clf1 = KNN(n_neighbors=2,useKDTree=False)
clf1.fit(X,y)
h = 0.1
cmap_light = ListedColormap(["green", "red"])
cmap_bold = ["darkorange", "c"]
x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1 , X[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf1.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
predikt = clf1.predict(X)

sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue = y,
        palette = cmap_bold)
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=cmap_light)

plt.show()
# ------------------------
h = 0.1
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


colors = ["blue", "green", "red"]
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
clf2 = KNN(7)
clf2.fit(X_r,y)
x_min, x_max = X_r[:, 0].min() - 1, X_r[:, 0].max() + 1
y_min, y_max = X_r[:, 1].min() - 1, X_r[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z)
kol1 = []
for i in range(2,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/i, random_state=0)
    clf3 = KNN(i, useKDTree=False)
    clf3.fit(X_train, y_train)
    prediction = clf3.predict(X_train)
    accuracy = clf3.score(prediction)
    kol1.append((i,accuracy))
print(kol1)




plt.show()

def KNN_regres_test():
    X, y = datasets.make_regression(
        n_samples=100,
        n_features=1,
        n_informative=2,
        n_targets=1,
        random_state=3,
        noise=10
    )
    print(X)
    knn_test=KNN(2,useKDTree=False)
    knn_test.fit(X,y)
    y_knn=knn_test.predict(X)
    print(y_knn)
    y_knn=y_knn.reshape(X.shape)
    X1=[]
    for i in range(len(y_knn)):
        X1.append([X[i],y_knn[i]])
    X1.sort()
    xx=[]
    yy=[]
    for x in X1:
        xx.append(x[0])
        yy.append(x[1])

    plt.scatter(X, y, color='green')
    plt.plot(xx, yy, color='red')
    plt.show()
KNN_regres_test()

X, y = datasets.make_regression(
        n_samples=100,
        n_features=1,
        n_informative=2,
        n_targets=1,
        random_state=93,
        noise=0
    )
k_val=[2,3,4,5,6,7,8,9,10,12]
score=[]

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data)

for k in k_val:
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,random_state=0)
    knn_test=KNN1(k,False)
    knn_test.fit(X_train,y_train)
    y_knn=knn_test.predict(X_test)
    score.append(knn_test.score(y_knn,y_test))
print(score)

from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
iris = datasets.load_iris()
X = iris.data
y = iris.target
error = []
loo = LeaveOneOut()
loo.get_n_splits(X)
for i in range(1, 41):

    classifier = KNeighborsClassifier(n_neighbors=i)
    y_pred = cross_val_predict(classifier, X, y, cv=loo)
    error.append(np.mean(y_pred != y))
error = np.asarray(error)
error.reshape((10,4))
print(error)
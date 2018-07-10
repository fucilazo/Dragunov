import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
pca_2c = PCA(n_components=2)
X_pca_2c = pca_2c.fit_transform(iris.data)
print(X_pca_2c.shape)
plt.scatter(X_pca_2c[:, 0], X_pca_2c[:, 1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print(pca_2c.explained_variance_ratio_.sum())

# 有时候PCA方法并不是足够有效，一个可行的方案是对信号进行白化
pca_2cw = PCA(n_components=2, whiten=True)
X_pca_1cw = pca_2cw.fit_transform(iris.data)
plt.scatter(X_pca_1cw[:, 0], X_pca_1cw[:, 1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
print(pca_2cw.explained_variance_ratio_.sum())
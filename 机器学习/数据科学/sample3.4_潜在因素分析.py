import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import FactorAnalysis

iris = datasets.load_iris()
fact_2c = FactorAnalysis(n_components=2)
X_factor = fact_2c.fit_transform(iris.data)
plt.scatter(X_factor[:, 0], X_factor[:, 1], c=iris.target, alpha=0.8, edgecolors='none')
plt.show()
"""
总体思路与PCA方法相似，但是它不需要对输入信号进行正交分解，因而也没有输出基
一般应用于系统中由一个潜在因素或结构的情形
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

colors = list()
palette = {0: "red", 1: "green", 2: "blue"}

for c in np.nditer(iris.target):
    colors.append(palette[int(c)]) 
dataframe = pd.DataFrame(iris.data, columns=iris.feature_names)
scatterplot = pd.plotting.scatter_matrix(dataframe, alpha=0.3, figsize=(10, 10),
                                         diagonal='hist', color=colors, marker='o', grid=True)
plt.show()

# 将pandas的DataFrame转换成一对包含数据和目标值的NumPy数组，通过几个命令就能完成这一任务
iris_data = iris.data[:, :4]
iris_target, iris_target_labels = pd.factorize(iris.target)
print(iris_data.shape, iris_target.shape)
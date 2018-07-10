import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
cov_data = np.corrcoef(iris.data.T)
print(iris.feature_names)
print(cov_data)

img = plt.matshow(cov_data, cmap=plt.cm.winter)
plt.colorbar(img, ticks=[-1, 0, 1])
plt.show()
"""
从图中可以看出，对角线上的数值为1，这是因为使用了协方差矩阵的标准化（对特征能量进行归一化）。
还可以发现第一和第三、第一和第四、以及第三和第四特征之间具有高度的相关性。
我们看到只有第二个特征几乎是独立的，而其他的特征则彼此相关。
我们现在有一个粗略的概念，约简数据集的潜在特征数量应该为2
"""
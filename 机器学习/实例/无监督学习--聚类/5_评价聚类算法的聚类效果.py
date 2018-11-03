"""
度量聚类算法的一个好的方法是观察集群被分离的离散程度。我们采用一个被称为轮廓系数得分的指标：
                            得分 = (x - y) / max(x, y)
其中，x表示在同一个集群中某个数据点与其他数据点的平均距离，y表示某个数据点与最近的另一个集群的所有点的平均距离
"""
import numpy as np
import matplotlib.pyplot as plt
import utilities
from sklearn import metrics
from sklearn.cluster import KMeans

data = utilities.load_data('data_perf.txt')

# 为了确定集群的最佳数列，我们迭代一系列的值，找出其中的峰值
scores = []
range_values = np.arange(2, 10)

for i in range_values:
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data)
    score = metrics.silhouette_score(data, kmeans.labels_, metric='euclidean', sample_size=len(data))

    print('Number of clusters = ', i)
    print('Silhouette score =', score)

    scores.append(score)

# 画出得分条形图
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouse score vs number of clusters')

# 画出数据
plt.figure()
plt.scatter(data[:, 0], data[:, 1], color='k', s=30, marker='o', facecolors='none')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

"""
由条形图可以看出，5个集群是最好的配置
"""
import numpy as np
import matplotlib.pyplot as plt
import utilities
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

X = utilities.load_data('data_multivar.txt')
# 带宽参数
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))
# 用meanshift计算聚类
meanshift_estimate = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimate.fit(X)
# 提取标记
labels = meanshift_estimate.labels_
# 从模型中提取集群的中心点
centroids = meanshift_estimate.cluster_centers_
num_clusters = len(np.unique(labels))
print('Number of clusters in input data = ', num_clusters)

plt.figure()
markers = '.*xv'

for i, markers in zip(range(num_clusters), markers):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], marker=markers, color='k')
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker='o', markerfacecolor='k', markeredgecolor='k', markersize=15)

plt.title('Cluster and their centroids')
plt.show()
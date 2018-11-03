import numpy as np
import matplotlib.pyplot as plt
import utilities
from sklearn import metrics
from sklearn.cluster import KMeans

data = utilities.load_data('data_multivar.txt')
num_clusters = 4

plt.figure()
plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)
step_size = 0.01
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_value, y_value = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())  # 消除横坐标
plt.yticks(())  # 消除纵坐标


# 初始化k-means对象，然后训练它
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(data)

# 预测网格中所有点的标记
predicted_labels = kmeans.predict(np.c_[x_value.ravel(), y_value.ravel()])

# 画出结果
predicted_labels = predicted_labels.reshape(x_value.shape)
plt.figure()
plt.clf()
plt.imshow(predicted_labels, interpolation='nearest',
           extent=(x_value.min(), x_value.max(), y_value.min(), y_value.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.scatter(data[:, 0], data[:, 1], marker='o', facecolors='none', edgecolors='k', s=30)

# 把中心点画在图形上
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, linewidths=3, color='k', zorder=10, facecolors='black')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())  # 消除横坐标
plt.yticks(())  # 消除纵坐标
plt.show()
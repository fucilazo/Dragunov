from itertools import cycle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt

input_file = 'data_perf.txt'
X = load_data(input_file)

# 寻找最优的epsilon参数值
eps_grid = np.linspace(0.3, 1.2, num=10)
silhouette_scores = []
eps_best = eps_grid[0]
silhouette_scores_max = -1
model_best = None
labels_best = None

# 搜索参数空间
for eps in eps_grid:
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = model.labels_
    # 提取性能指标
    silhouette_score = round(metrics.silhouette_score(X, labels), 4)
    silhouette_scores.append(silhouette_score)
    print('Epsilon:', eps, ' --> silhouette score:', silhouette_score)
    # 保存指标的最佳得分和对应的epsilon值
    if silhouette_score_max > silhouette_score:
        silhouette_scores_max = silhouette_score
        eps_best = eps
        model_best = model
        labels_best = labels

plt.figure()
plt.bar(eps_grid, silhouette_scores, width=0.05, color='k', align='center')
plt.title('Silhouette score vs epsilon')
# 打印最优参数
print('Best epsilon =', eps_best)

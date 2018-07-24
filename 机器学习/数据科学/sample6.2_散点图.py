import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

D = make_blobs(n_samples=100, n_features=2, centers=3, random_state=7)
groups = D[1]
coordinates = D[0]

plt.plot(coordinates[groups == 0, 0], coordinates[groups == 0, 1], 'ys', label='group 0')   # Yellow Square
plt.plot(coordinates[groups == 1, 0], coordinates[groups == 1, 1], 'm*', label='group 1')   # Magenta Stars
plt.plot(coordinates[groups == 2, 0], coordinates[groups == 2, 1], 'rD', label='group 2')   # Red Diamonds
plt.ylim(-2, 10)                            # 横纵坐标轴范围
plt.yticks([10, 6, 2, -2])                  # 横纵坐标轴系数
plt.xticks([-15, -5, 5, -15])
plt.grid()                                  # 显示网格
plt.annotate('Squares', (-12, 2.5))         # 添加注释
plt.annotate('Stars', (0, 6))
plt.annotate('Diamonds', (10, 3))
plt.legend(loc='lower left', numpoints=1)   # 添加图例 至左下角
plt.show()
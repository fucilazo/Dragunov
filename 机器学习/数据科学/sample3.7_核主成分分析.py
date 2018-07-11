import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA


# 假设有一组由如下代码生成的双圆环数据集
def circular_points(radius, N):
    return np.array([[np.cos(2*np.pi*t/N)*radius, np.sin(2*np.pi*t/N)*radius] for t in range(N)])


N_point = 50
fake_circular_data = np.vstack([circular_points(1.0, N_point), circular_points(5.0, N_point)])
fake_circular_data += np.random.rand(*fake_circular_data.shape)
fake_circular_target = np.array([0]*N_point + [1]*N_point)
plt.scatter(fake_circular_data[:, 0], fake_circular_data[:, 1], c=fake_circular_target, alpha=0.8, edgecolors='none')
plt.show()

# 由于输入的是圆形数据，所有的线性变换方法都无法实现将图中的两种颜色的点分离
kpca_2c = KernelPCA(n_components=2, kernel='rbf')
X_kpca_2c = kpca_2c.fit_transform(fake_circular_data)
plt.scatter(X_kpca_2c[:, 0], X_kpca_2c[:, 1], c=fake_circular_target, alpha=0.8, edgecolors='none')
plt.show()  # 之后仅需要使用线性技术就可以处理这个数据了
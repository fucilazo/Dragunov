"""
样本中显著脱离其他数值的数据称为异常值（Outlier），其他预期的观测值标记为正常值或内点（Inlier）。
主要由三种原因：
1.极少发生的事件。常用的方法是将这样的点去除或降低权重，另一种方法是增加样本数量
2.经常发生的另一种分布，可认为发生了影响样本生成的错误。这样的数值必须去除
3.数据点明显是某种类型的错误。用均值或常见的类来替换异常点，否则直接删除
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn import svm

# 单变量异常检测
boston = load_boston()
continuous_variables = [n for n in range(np.shape(boston.data)[1]) if n != 3]   # 该数据集的说明指出，索引号为3的变量为二进制，所以不应该用该变量来检测异常数据
print('用来检测异常的索引值：', continuous_variables)
normalized_data = preprocessing.StandardScaler().fit_transform(boston.data[:, continuous_variables])    # 将数据规范为均值为零、方差为1的数据
outliers_rows, outliers_columns = np.where(np.abs(normalized_data) > 3)     # 找出绝对值大于3倍标准差的值
print('可疑异常值的坐标：\n', list(zip(outliers_rows, outliers_columns)))   # 输出可疑异常值的行、列坐标
"""
单变量方法可以检测出相当多的潜在异常值，但是它不能检测那些不是极端值的异常值。为了发现这种情况，可以使用降维算法
"""

# EllipticEnvelope
blobs = 1
blob = make_blobs(n_samples=100, n_features=2, centers=blobs, cluster_std=1.5, shuffle=True, random_state=5)    # 创建100个样本的分布
robust_covariance_eat = EllipticEnvelope(contamination=.1).fit(blob[0])     # 运行污染参数为10%的EllipticEnvelope函数，找出分布中极端的数值，再使用.fit()方法展开首次适应
detection = robust_covariance_eat.predict(blob[0])  # 利用.predict方法在适应后的数据上进行预测
# 以上结果得到的是1和-1的向量
outliers = np.where(detection == -1)
inlers = np.where(detection == 1)
plt.subplot(121)
plt.plot(blob[0][:, 0], blob[0][:, 1], 'x', markersize=10, color='black', alpha=0.8)
plt.subplot(122)
a = plt.plot(blob[0][inlers, 0], blob[0][inlers, 1], 'x', markersize=10, color='black', alpha=0.8, label='inliers')
b = plt.plot(blob[0][outliers, 0], blob[0][outliers, 1], 'o', markersize=6, color='black', alpha=0.8, label='outliers')
plt.legend((a[0], b[0]), ('inliers', 'outliers'), numpoints=1, loc='lower right')
plt.show()

# OneClassSVM
pca = PCA(n_components=5)
Zscore_component = pca.fit_transform(normalized_data)
vtot = 'PCA Variance explained ' + str(round(np.sum(pca.explained_variance_ratio_), 3))
outliers_fraction = 0.5     # 异常数据概率
nu_estimate = 0.95 * outliers_fraction + 0.05   # 数量计量核心参数nu_estimate,gamma
machine_learning = svm.OneClassSVM(kernel='rbf', gamma=1.0/len(normalized_data), degree=3, nu=nu_estimate)
machine_learning.fit(normalized_data)
detection = machine_learning.predict(normalized_data)
outliers = np.where(detection == -1)
regular = np.where(detection == 1)
for r in range(1, 5):
    a = plt.plot(Zscore_component[regular, 0], Zscore_component[regular, r], 'x', markersize=2, color='blue', alpha=0.6, label='inliers')
    b = plt.plot(Zscore_component[outliers, 0], Zscore_component[outliers, r], 'o', markersize=6, color='red', alpha=0.8, label='outliers')
    plt.xlabel('Component(' + str(round(pca.explained_variance_ratio_[0], 3)) + ')')
    plt.ylabel('Component' + str(r+1) + '(' + str(round(nu_estimate)) + ')')
    plt.xlim([-7, 7])
    plt.ylim([-6, 6])
plt.legend((a[0], b[0]), ('inliers', 'outliers'), numpoints=1, loc='best')
plt.title(vtot)
plt.show()

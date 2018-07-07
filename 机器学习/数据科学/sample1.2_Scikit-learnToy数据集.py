from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# print("提供数据集的总体描述：")
# print(iris.DESCR)
# print("所有的数据：")
# print(iris.data)
# print("数据数量，特征数量：")
# print(iris.data.shape)
# print("描述特征的名称：")
# print(iris.feature_names)
# print("每一个样本对应的类型：")
# print(iris.target)
# print("描述数量：")
# print(iris.target.shape)
# print("记录目标中的类别名称：")
# print(iris.target_names)


X = iris.data[:, :2]    # 所有的行取前两列
# print(X)
# print(X[:, 0], '----',  X[:, 1])
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()

# 使用不同颜色形状标识不同类别的花
Y = iris.target
print(Y)
for i, colors, marker in [(0, 'red', 'o'), (1, 'blue', '+'), (2, 'green', 'x')]:
    plt.scatter(X[Y == i, 0], X[Y == i, 1], color=colors, marker=marker)
plt.show()

X = iris.data[:, 2:]    # 使用另外两个维度
for i, colors, marker in [(0, 'red', 'o'), (1, 'blue', '+'), (2, 'green', 'x')]:
    plt.scatter(X[Y == i, 0], X[Y == i, 1], color=colors, marker=marker)
plt.show()
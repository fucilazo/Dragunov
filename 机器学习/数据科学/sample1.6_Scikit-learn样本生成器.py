from sklearn import datasets
import time

time.clock()
X, Y = datasets.make_classification(n_samples=10**6, n_features=10, random_state=101)   # 100w个样本
print(X.shape, Y.shape)
print(time.clock())

print(datasets.make_classification(1, n_features=4, random_state=101))  # 不同的计算机下此条代码结果不变
print(time.clock())
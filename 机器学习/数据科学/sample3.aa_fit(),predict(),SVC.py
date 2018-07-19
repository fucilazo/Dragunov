import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # 数据
y = np.array([1, 1, 2, 2])  # 数据对应的标签
clf = SVC()     # 创建分类器对象
clf.fit(X, y)   # 用训练数据拟合分类器模型
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto',
    kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1]]))   # 用训练好的分类器去预测[-0.8, -1]的标签
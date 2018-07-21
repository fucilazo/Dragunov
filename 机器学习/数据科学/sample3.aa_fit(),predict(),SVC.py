import numpy as np
from sklearn.svm import SVC
from sklearn import metrics

# SVM二分类
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])  # 数据
y = np.array([1, 1, 2, 2])  # 数据对应的标签
clf = SVC()     # 创建分类器对象
clf.fit(X, y)   # 用训练数据拟合分类器模型
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto',
    kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
print(clf.predict([[-0.8, -1]]))   # 用训练好的分类器去预测[-0.8, -1]的标签
print('-----------------------------------------------------')


# SVM多分类
X = [[0], [1], [2], [3]]
Y = [0, 1, 2, 3]
clf = SVC(decision_function_shape='ovo')
clf.fit(X, Y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
dec = clf.decision_function([[1]])  # 返回样本距离超平面的距离，对于多分类，得到每对分类器的输出
print(dec)
print(dec.shape[1])     # 对应的分类器为[AB,AC,AD,BC,BD,CD]-->根据结果正负得到每对分类器结果为[B B A B B C]-->B出现最多所以为预测label
print(clf.predict([[1]]))   # -->B -->[1]

clf.decision_function_shape = "ovr"     # 而ovr直接选取绝对值最大的那个作为分类结果
dec = clf.decision_function([[1]])
print(dec)
print(dec.shape[1])
print(clf.predict([[1]]))

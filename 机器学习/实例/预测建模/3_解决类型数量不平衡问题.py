import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import utilities

input_file = 'data_multivar_imbalance.txt'
X, y = utilities.load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

plt.scatter(class_0[:, 0], class_0[:, 1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolors='None', edgecolors='black', marker='s')
plt.show()

# 用线性核函数建立一个SVM分类器
params = {'kernel': 'linear'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
target_names = ['Class-0', 'Class-1']
print('#' * 60)
print('Classifier performance on training daraset')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#' * 60)
"""
可以发现没有边界线，因为分类器不能区分两种类型，导致Class-0的准确性是 0%
"""
# 调整权重
params = {'kernel': 'linear', 'class_weight': 'balanced'}   # 参数class_weight的作用是统计不同类型的数据点的数量，调整权重
classifier = SVC(**params)
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
target_names = ['Class-0', 'Class-1']
print('#' * 60)
print('Classifier performance on training daraset')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#' * 60)
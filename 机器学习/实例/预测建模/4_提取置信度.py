import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import utilities

input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)

# 测量数据点与边界的距离
input_datapoint = np.array([[2, 1.5], [8, 9], [4.8, 5.2], [4, 4], [2.5, 7], [7.6, 2], [5.4, 5.9]])
print('Distance from the boundary')
for i in input_datapoint:
    print(i, ' --> ', classifier.decision_function(i.reshape(1, -1))[0])
"""
到边界点的距离向我们提供了一些关于数据点的信息，但是它并不能准确的告诉我们分类器能够输入某个类型的置信度有多大
为了解决这个问题，需要用到概率输出，这是一种将不用类别的距离度量转换成概率度量的方法
"""
# 测量置信度
params = {'kernel': 'rbf', 'probability': True}     # 参数probability告诉SVM训练的时候要计算出概率
classifier = SVC(**params)
classifier.fit(X_train, y_train)
print('Confidence measure')
for i in input_datapoint:
    print(i, ' --> ', classifier.predict_proba(i.reshape(1, -1))[0])

# 显示数据点与边界的位置
utilities.plot_classifier(classifier, input_datapoint, [0] * len(input_datapoint), 'Input datapoints', True)
plt.show()


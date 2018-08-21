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

# 使用三次多项式方程，方程次数越高，曲线弯曲程度越大。但训练时间越长，因为计算强度更大
params = {'kernel': 'poly', 'degree': 3}
classifier = SVC(**params)
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
target_names = ['Class-0', 'Class-1']
print('#' * 60)
print('Classifier performance on training daraset')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#' * 60)

# 使用径向基函数建立非线性分类器
params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
target_names = ['Class-0', 'Class-1']
print('#' * 60)
print('Classifier performance on training daraset')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#' * 60)
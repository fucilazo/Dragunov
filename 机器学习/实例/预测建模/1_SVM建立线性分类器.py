"""
SVM是用来构建分类器和回归器的监督学习模型。SVM通过对数学放方程组求解，可以找出两组数据之间的最佳分割边界。

SVM的主要思想可以概括为两点：
    1.它是针对线性可分情况进行分析，对于线性不可分的情况，通过使用非线性映射算法将低维输入空间线性不可分的样本转化为高维特征空间使其线性可分，从而使
      得高维特征空间采用线性算法对样本的非线性特征进行线性分析成为可能。
    2.它基于结构风险最小化理论之上在特征空间中构建最优超平面，使得学习器得到全局最优化，并且在整个样本空间的期望以某个概率满足一定上界。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import utilities    # 自建库

input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

# 对数据分类
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()
"""
 -------------------------------
|目标就是将实心方块和空心方块分离开来|
 -------------------------------
"""
# 分割数据集并用SVM训练模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
# 用线性核函数（linear kernel）初始化一个SVM对象
params = {'kernel': 'linear'}
classifier = SVC(**params)
# 训练线性SVM分类器
classifier.fit(X_train, y_train)
# 显示分类结果
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

# 分类器对测试数据分类
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()

# 计算训练数据集的准确性
target_names = ['Class-0', 'Class-1']
print('#' * 60)
print('Classifier performance on training daraset')
print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
print('#' * 60)

# 分类器为测试数据集生成的分类报告
y_test_pred = classifier.predict(X_test)
print('#' * 60)
print('Classification report on test dataset')
print(classification_report(y_test, y_test_pred, target_names=target_names))
print('#' * 60)
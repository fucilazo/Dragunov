from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target   # 0,1,2...8,9,0
print(X[0])         # 手写数字集的灰度图
# 加载三个不同的机器学习假设和用于分类的支持向量机
h1 = svm.LinearSVC(C=1.0)   # linear SVC
h2 = svm.SVC(kernel='rbf', degree=3, gamma=0.001, C=1.0)    # Radial basis SVC
h3 = svm.SVC(kernel='poly', degree=3, C=1.0)    # 3rd degree polynomial SVC
# 对数据进行线性SVC拟合
h1.fit(X, y)            # 使用数组X拟合一个模型，以正确预测出y向量中的十个类别之一
print(h1.score(X, y))   # 用预测值相对y向量真值的平均精度来评价模型性能
print('--------------------------')
"""
由于预测过程变得随机，记忆效应导致对没见过的数据进行预测时变化较大，有三种可行的解决方案：
1.增加样本数量
2.使用简单的机器学习算法，这样算法既不倾向于记忆，也不会过分拟合于数据背后的复杂规则
3.使用正则化方法限制非常复杂的模型，使算法对有些变量降低权重，甚至去除一定量的变量
"""

# 一个好的替代方法是将原始数据分为训练集（70%~80%）和测试集（20%~30%）
chosen_random_state = 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=chosen_random_state)
h1.fit(X_train, y_train)    # 用训练数据拟合分类器模型
print(h1.score(X_test, y_test))
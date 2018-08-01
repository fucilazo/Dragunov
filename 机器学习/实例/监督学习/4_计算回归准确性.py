"""
回归器可以用许多不同的指标进行衡量，部分指标如下所示：
◎平均值绝对误差（mean absolute error）：数据集所有数据点的绝对误差平均值
◎均方误差（mean squared error）：数据集的所有数据点的误差的平方的平均数
◎中位数误差（median absolute error）：数据集所有数据点的误差的中位数。主要优点是可以消除异常值的干扰，单个坏点不会影响结果
◎解释方差分（explained variance score）：用于衡量模型对数据集波动的解释能力。1分表示模型是完美的
◎R方得分（R2 score）：指确定性相关系数，用于衡量模型对未知样本预测的效果
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

file_name = 'data_singlevar.txt'
X = []
y = []
with open(file_name, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)


num_training = int(0.8 * len(X))
num_test = len(X) - num_training
X_train = np.array(X[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])
X_test = np.array(X[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_train_pred = linear_regressor.predict(X_train)
y_test_pred = linear_regressor.predict(X_test)

# 计算指标
print('平均值绝对误差\t', round(metrics.mean_absolute_error(y_test, y_test_pred), 2))
print('均方误差\t\t\t', round(metrics.mean_squared_error(y_test, y_test_pred), 2))
print('中位数误差\t\t', round(metrics.median_absolute_error(y_test, y_test_pred), 2))
print('解释方差分\t\t', round(metrics.explained_variance_score(y_test, y_test_pred), 2))
print('R方得分\t\t\t', round(metrics.r2_score(y_test, y_test_pred), 2))
"""
通常选用一两个指标来评估我们的模型，通常的做法是尽量保证均方误差最低，而且解释方差分最高
"""
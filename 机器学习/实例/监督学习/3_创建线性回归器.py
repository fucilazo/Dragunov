"""
线性回归的目标是提取输入变量与输出变量的关联现行模型，这就要求实际输出与线性方程预测的蔬菜的残差平方和最小化，这种方法被称为普通最小二乘法

用一条曲线对这些点进行拟合效果会更好，但是线性回归不允许这样做。线性回归的主要优点是方程简单
如果想采用非线性回归，可能会得到更精确的模型，但是拟合速度会慢很多
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

file_name = 'data_singlevar.txt'
X = []
y = []
with open(file_name, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

# 把数据分成训练集和测试集
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
# 训练数据
X_train = np.array(X[:num_training]).reshape((num_training, 1))     # 将数据竖向排列
y_train = np.array(y[:num_training])
# 测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])
# 创建线性回归对象
linear_regressor = linear_model.LinearRegression()
# 用训练集训练模型
linear_regressor.fit(X_train, y_train)

# 查看拟合效果
y_train_pred = linear_regressor.predict(X_train)    # 返回预测值（使用训练好的模型对X_train计算预测值，返回线性数据）
plt.figure()
plt.xlim(-8, 6)
plt.ylim(0, 6)
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')

# 用测试集进行预测
y_test_pred = linear_regressor.predict(X_test)
plt.figure()
plt.xlim(-8, 6)
plt.ylim(0, 6)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()
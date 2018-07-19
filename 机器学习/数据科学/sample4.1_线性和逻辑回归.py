import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 线性回归
# 通过训练/测试划分交叉验证将数据集划分为两部分（在这个例子中，80%用于训练，20%用于测试）
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=0)
# 在训练集上训练拟合回归模型，并在测试集上预测目标变量。通过MAE（平均绝对误差）分值衡量回归任务的精度
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('MAE', mean_absolute_error(y_test, y_pred))

# 逻辑回归
# 猜测房价是在我们感兴趣的阈值上面还是下面（即从回归问题转到一个二值分类问题）
# 准备数据集
avg_price_house = np.average(boston.target)
high_priced_idx = (y_train >= avg_price_house)
y_train[high_priced_idx] = 1
y_train[np.logical_not(high_priced_idx)] = 0
y_train = y_train.astype(np.int8)
high_priced_idx = (y_test >= avg_price_house)
y_test[high_priced_idx] = 1
y_test[np.logical_not(high_priced_idx)] = 0
y_test = y_test.astype(np.int8)
# 训练和应用分类器，为了衡量它的性能，我们将简单地输出分类报告
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
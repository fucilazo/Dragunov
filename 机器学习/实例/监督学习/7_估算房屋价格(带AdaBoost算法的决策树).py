"""
决策树是一个树状模型，每个节点都做出一个决策，从而影响最终结果。叶子节点表示输出数值，分支表示根据输入特征做出的中间决策
AdaBoost算法是指自适应增强算法，是一种利用其他系统增强模型准确性的技术。是将不同版本的算法结果进行组合，用加权汇总的方式获得最终结果，称为弱学习器
AdaBoost算法在每个阶段获取的信息都会反馈到模型中，这样学习器就可以在后一阶段重点训练难以分类的样本
首先使用AdaBoost算法对数据集进行回归拟合，再计算误差，然后根据误差评估结果，用同样的数据集重新拟合，直到达到预期的准确性
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

housing_data = load_boston()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)  # 将数据的顺序打乱
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
# 拟合决策树回归模型
dt_regressor = DecisionTreeRegressor(max_depth=4)   # 选一个最大深度为4的决策树
dt_regressor.fit(X_train, y_train)
# 再用AdaBoost算法的决策树回归模型进行拟合
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)

# 评价决策树回归器的训练效果
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)
print('### 决策树学习效果 ###')
print('Mean squared error = ', round(mse, 2))
print('Explained variance score', round(evs, 2))

# 评价AdaBoost算法改善效果
y_pred_ab = ab_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)
print('### AdaBoost算法改善效果 ###')
print('Mean squared error = ', round(mse, 2))
print('Explained variance score', round(evs, 2))
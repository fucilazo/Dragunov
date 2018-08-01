import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rt'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])
    # 提取特征名称
    feature_names = np.array(X[0])
    # 将第一行特征名称移除，仅保留数值
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names


def plot_feature_importances(feature_importances, title, feature_names):
    feature_importances = 100.0 * (feature_importances / max(feature_importances))  # 将重要性值标准化
    index_sorted = np.flipud(np.argsort(feature_importances))   # 将得分从高到低排列
    pos = np.arange(index_sorted.shape[0]) + 0.5    # 让x轴上的标签居中显示
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()


# 读取数据并打乱
X, y, feature_names = load_dataset('bike_day.csv')
X, y = shuffle(X, y, random_state=7)

num_train = int(0.9 * len(X))
X_train, y_train = X[:num_train], y[:num_train]
X_test, y_test = X[num_train:], y[num_train:]

# 训练回归器（参数n_estimators表示随机森林需要使用的决策数数量；参数max_depth指每个决策树的最大深度；
#           参数min_samples_split指决策树分裂一个节点需要用到的最小数据样本量）
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=2)
rf_regressor.fit(X_train, y_train)

# 评价随机森林的回归器训练效果
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print('### 随机森林回归器 ###')
print('Mean squared error = ', round(mse, 2))
print('Explained variance score', round(evs, 2))

plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)
"""
结果可以看出温度（temp）是自行车租赁的最重要因素
"""
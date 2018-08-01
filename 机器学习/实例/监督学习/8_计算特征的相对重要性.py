import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

housing_data = load_boston()
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(X_train, y_train)
ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(X_train, y_train)


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


plot_feature_importances(dt_regressor.feature_importances_, 'Decision tree regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, 'AdaBoost regressor', housing_data.feature_names)
"""
由结果可以发现，不带AdaBoost算法的决策树回归器显示的最重要特征是RM
加入AdaBoost算法之后，最重要特征是LSTAT，表明AdaBoost算法对决策树回归器的训练效果有所改善
"""
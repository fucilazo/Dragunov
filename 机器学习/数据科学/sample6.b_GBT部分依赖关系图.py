import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor
"""
对特征重要性的估计能帮助你做出最佳的特征选择，以确定模型中该使用哪种特征
梯度提升树（GBT）方法通过控制分析中所有的其他变量的影响，提供了一个清晰的观测变量与预测结果关系的视图
这些信息有助于对因果关系动力学的理解，能够提供比使用非常有效的探索性数据分析方法更深入的见解
"""
boston = load_boston()
X, y = boston.data, boston.target
feature_names = np.array([' '.join([str(b), a]) for a, b in zip(boston.feature_names, range(len(boston.feature_names)))])
GBM = GradientBoostingRegressor(n_estimators=100, random_state=101).fit(X, y)
features = [5, 12, (5, 12)]
fig, axis = plot_partial_dependence(GBM, X, features, feature_names=feature_names)
plt.show()
"""
当你制定好分析计划后，plot_partial_dependence类会自动提供可视化方法。你需要提供一系列特征的索引及其元组，这些特征和元组可以单独地绘制到热图上

在之前的例子中，对房间平均数和低层社会人口的比例已经进行了描述，从而展现了预期的行为。
热图解释了这两个变量是如何共同作用于结果数值的，事实表明它们并不以任何特定的方式相互影响
然而，它也表明：当LSTAT大于5时，变量LSTAT时房价结果数值的“指示器”
"""
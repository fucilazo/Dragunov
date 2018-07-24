import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
"""
第三章可以得出结论，选择合适的变量可以改进学习过程，例如减少学习中的噪声、方差估计和巨大的计算负荷。
集成方法————特别时随机森林方法，可以提供一个不同的视角，来认识变量与数据集中的其他变量一起工作时所承担的角色，与之对比的是采用后向或前向逐步选择变量的贪婪算法
"""
boston = load_boston()
X, y = boston.data, boston.target
feature_names = np.array([' '.join([str(b), a]) for a, b in zip(boston.feature_names, range(len(boston.feature_names)))])
RF = RandomForestRegressor(n_estimators=100, random_state=101).fit(X, y)
importance = np.mean([tree.feature_importances_ for tree in RF.estimators_], axis=0)
std = np.std([tree.feature_importances_ for tree in RF.estimators_], axis=0)
indices = np.argsort(importance)
range_ = range(len(importance))
plt.figure()
plt.title('Random Forest importance')
plt.barh(range_, importance[indices], color='r', xerr=std[indices], alpha=0.4, align='center')
plt.yticks(range(len(importance)), feature_names[indices])
plt.ylim([-1, len(importance)])
plt.xlim([0.0, 0.65])
plt.show()
"""
对于每一个估计器（在本例中有100个），算法都估计一个得分以对每个变量的重要性进行排序。随机森林模型是由包含许多分枝的复杂的决策树组成的。
如果随意将某个变量的原始数值进行置换，置换模型与原始模型的预测精准性差别较大，则认为这个变量是重要的
在本例LSTAT分析中，一个地区的底层社会人口的百分比与每户平均房间数（RM）是随机森林模型最关键的变量
"""
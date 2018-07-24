import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
groups = list(iris.target)
iris_df['groups'] = pd.Series([iris.target_names[k] for k in groups])   # 在下面的所有示例中，将使用数据框iris_df

# 箱线图描绘了分布的主要特征
boxplots = iris_df.boxplot(return_type='axes')
plt.show()
# 分割数据
boxplots2 = iris_df.boxplot(column='sepal length (cm)', by='groups', return_type='axes')
plt.show()
# 通过密度图和直方图能够刻画出分布是否有峰或谷
densityplot = iris_df.plot(kind='density')
plt.show()
single_distribution = iris_df['petal width (cm)'].plot(kind='hist', alpha=0.5)
plt.show()

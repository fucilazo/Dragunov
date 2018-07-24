import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix
from pandas.plotting import parallel_coordinates

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
groups = list(iris.target)
iris_df['groups'] = pd.Series([iris.target_names[k] for k in groups])   # 在下面的所有示例中，将使用数据框iris_df

# 散点图
colors_palette = {0: 'red', 1: 'yellow', 2: 'blue'}
colors = [colors_palette[c] for c in groups]
simple_scatterplot = iris_df.plot(kind='scatter', x=0, y=1, c=colors)
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.show()

# 蜂箱图
hexbin = iris_df.plot(kind='hexbin', x=0, y=1, gridsize=10)
plt.grid(color='gray', linestyle='--', alpha=0.5)
plt.show()

# 散点图矩阵
matrix_of_scatterplots = scatter_matrix(iris_df, alpha=0.2, figsize=(6, 6), color=colors, diagonal='kde')
plt.show()

# 平行坐标
pll = parallel_coordinates(iris_df, 'groups')
plt.show()

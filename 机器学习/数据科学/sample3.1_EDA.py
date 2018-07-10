import pandas as pd
import matplotlib.pyplot as plt

iris_filename = 'iris.csv'
iris = pd.read_csv(iris_filename, header=None, names=['sepal_length', 'sepal_width',
                                                      'petal_length', 'petal_width', 'target'])
print(iris.head())      # 加载原数据
print(iris.describe())  # 得到数值特征：观测值数量、各特征的平均值、标准差、最大值、最小值，以及一些百分位数（25%,50%,75%）
print(iris.quantile([0.1, 0.9]))    # 查看10%和90%的两个分位数的结果
iris.boxplot()
plt.show()

# 创建共生矩阵，查看特征之间的关系
crosstab = pd.crosstab(iris['petal_length'] > iris['petal_length'].mean(),
                       iris['petal_width'] > iris['petal_width'].mean())
print(crosstab)     # 由结果看出，上述两种特征与均值的比较几乎是同时发生，可以假设这两个事件之间由很强的关联
# 使用图形化显示它们之间的关系：
plt.scatter(iris['petal_width'], iris['petal_length'], alpha=1.0, color='k')
plt.xlabel('petal width')
plt.ylabel('petal length')
plt.show()  # 通过图中的趋势可以推断出两个坐标变量x和y是密切相关的
# 最后进行特征分布的检验，使用直方图近似表示特征的概率分布
plt.hist(iris['petal_width'], bins=20)  # 选择20个分箱。根据经验法则，分箱数量是观测数量的平方根，然后不断修正直到得到一个很好的概率分布
plt.xlabel('petal width distribution')
plt.show()
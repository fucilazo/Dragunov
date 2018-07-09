import pandas as pd

iris_filename = 'iris.csv'
iris = pd.read_csv(iris_filename, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
# 生成掩模，用来选出花萼长度大于6.0的数据
mask_feature = iris['sepal_length'] > 6.0
print(mask_feature)

# 假如 现在要用“New Label”标签替换类别（target）中的“virginica”标签
mask_target = iris['target'] == 'virginica'
iris.loc[mask_target, 'target'] = 'New Label'
# 仅查看target标签列中的种类
print(iris['target'].unique())

# 了解每个特征的统计数据
group_target_mean = iris.groupby(['target']).mean()
print(group_target_mean)
group_target_var = iris.groupby(['target']).var()
print(group_target_var)

iris_sort = iris.sort_values(by='sepal_length').head()   # 对观测值进行排序
print(iris_sort)
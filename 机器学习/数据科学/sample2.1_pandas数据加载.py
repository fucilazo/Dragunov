import pandas as pd

iris_filename = 'iris.csv'
iris = pd.read_csv(iris_filename, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target'])
print(iris.head())      # head(int)可设置显示几行
print("\n每列的名称：")
print(iris.columns)
print("\n提取target列：")
print(iris['target'])   # iris['xxx'][int]可以提取'xxx'列第几行
print("数据维数", iris['target'].shape)
print("\n索引列：")
print(iris[['sepal_length', 'sepal_width']])
print("数据维数", iris[['sepal_length', 'sepal_width']].shape)


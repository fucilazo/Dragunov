import pandas as pd

dataset = pd.read_csv('sample2.7.csv')
print(dataset)  # 表中的“n”列会视为特征列
dataset = pd.read_csv('sample2.7.csv', index_col=0)     # 将“n”列会视为索引列
print(dataset, '\n')

print(dataset['val3'][104])         # 提取索引号为104的val3
print(dataset.loc[104, 'val3'])     # 同上
print(dataset.ix[104, 'val3'])      # 同上，但ix()适用于各种类型的索引（标签或位置）
print(dataset.ix[104, 2])
print(dataset.iloc[4, 2])           # 使用行数和列数来指定数据元素

# 子矩阵检索：
print(dataset[['val3', 'val2']][0:2])
print(dataset.loc[range(100, 102), ['val3', 'val2']])
print(dataset.ix[range(100, 102), ['val3', 'val2']])
print(dataset.ix[range(100, 102), [2, 1]])
print(dataset.iloc[range(2), [2, 1]])
import numpy as np
import pandas as pd

# loadtxt
housing = np.loadtxt('xxxxxxx.csv', delimiter=',')
housing_int = np.loadtxt('xxxxxxx.csv', delimiter=',', dtype=int)
print(housing[0, :3], '\n', housing_int[0, :3])
# loadtxt函数默认制表符作为文件中数值的分隔符。如果分隔符是其他符号，则必须定义参数进行说明。
print(type(housing))
print(housing, shape)

# read_csv
iris_filename = 'xxxxx.csv'
iris = pd.read_csv(iris_filename, sep=',', decimal='.', header=None, names=['sepal_length', 'sepal_width',
                                                                            'petal_length', 'petal_width', 'target'])
# 指定分隔符sep  小数点表达方式deciaml  是否有标题行header    变量名称names
print(type(iris))

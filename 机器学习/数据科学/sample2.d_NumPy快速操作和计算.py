import numpy as np

a = np.arange(5).reshape(1, 5)
a += 1
a *= a
print(a, '\n')    # >>>[[ 1 4 9 16 25]]

# 矩阵互相运算
a = np.array([1, 2, 3, 4, 5] * 5).reshape(5, 5)
b = a.T
print(a*b, '\n')

# 按列、行求和    axis：0表示水平轴，1表示垂直轴
print(np.sum(a, axis=0))
print(np.sum(a, axis=1), '\n')

# NumPy切片与索引
M = np.arange(10*10, dtype=int).reshape(10, 10)
print(M)
print(M[2:9:2, :])      # [起始位置:结束位置:步长， 数据段]
print(M[2:9:2, 5:])
print(M[2:9:2, 5::-1])  # 从第6列逆序索引

row_index = (M[:, 0] >= 20) & (M[:, 0] <= 80)
col_index = M[0, :] >= 5
print(M[row_index, :][:, col_index])
print('-----------------------')

# 数组堆叠  h表示水平 v表示垂直
dataset = np.arange(10*5).reshape(10, 5)
single_line = np.arange(1*5).reshape(1, 5)
a_few_lines = np.arange(3*5).reshape(3, 5)
print(np.vstack((dataset, single_line)))    # 追加行
print(np.vstack((dataset, a_few_lines)))    # 追加矩阵
print('\n')
bias = np.ones(10).reshape(10, 1)   # 追加变量
print(np.hstack((dataset, bias)))
bias = np.ones(10)                  # 以偏差追加，偏差可以是任何与数组行数相同的数据序列
print(np.column_stack((dataset, bias)))
print('\n')
print(np.dstack((dataset*1, dataset*2, dataset*3)))   # 处理三维数组
print('-------------------------')

# 插入数据
print(np.insert(dataset, 3, bias, axis=1))
print(np.insert(dataset, 3, dataset.T, axis=1))
print(np.insert(dataset, 3, np.ones(5), axis=0))
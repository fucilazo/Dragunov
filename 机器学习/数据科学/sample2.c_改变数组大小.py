import numpy as np

original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
array_a = original_array.reshape(4, 2)          # 维数改为4x2
array_b = original_array.reshape(4, 2).copy()   # 同上，只是增加了copy()方法
array_c = original_array.reshape(2, 2, 2)       # 维数改为2x2x2
original_array[0] = -1
print(array_a, '\n-----------\n', array_b, '\n-----------\n', array_c, '\n-----------')
# 改变原始数组形状  同original_array.shape = (4, 2)
original_array.resize(4, 2)
print(original_array, '\n-----------')
# 实现行列交换，即矩阵转置
print(original_array.T)
print(original_array.transpose())
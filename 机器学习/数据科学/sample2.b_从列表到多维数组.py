import numpy as np

# 二维数组
a_lis_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
array_2D = np.array(a_lis_of_lists)
print(array_2D, '\n')

# 三维数组
a_lis_of_lists_of_lists = [[[1, 2], [3, 4], [5, 6]],
                           [[7, 8], [9, 10], [11, 12]]]
array_3D = np.array(a_lis_of_lists_of_lists)
print(array_3D)

import numpy as np
import pandas as pd

# 转换列表为一维数组
list_of_ints = [1, 2, 3]
array_1 = np.array(list_of_ints)
print(array_1)
print(type(array_1))
print(array_1.dtype)    # 输出为'int32'(取决于系统)，默认整数范围是从 -2^31~2^31-1

# 计算数组对象内存占用情况
print(array_1.nbytes)
# 手动设定数组占用内存
array_1 = np.array(list_of_ints, dtype='int8')
print(array_1.nbytes)
"""
=======  =====      ============================ 
类型      字节数      描述
bool       1        布尔类型
int_       4        默认为整型
int8       1        一个字节大小（-128~127）
int16      2        整型（-32768~32767）
int32      4        整型（-2^31~2^31-1）
int64      8        整型（-2^63~2^63-1）
uint8      1        无符号型（0~255）
uint16     2        无符号型（0~65535）
uint32     3        无符号型（0~2^31-1）
uint64     4        无符号型（0~2^64-1）
float_     8        float64的简写形式
float16    2        半精度浮点数（指数5位，尾数10位）
float32    4        单精度浮点数（指数8位，尾数23位）
float64    8        双精度浮点数（指数11位，尾数52位）
=======  =====      ============================
"""

# 异构列表
complex_list = [1, 2, 3] + [1., 2., 3.] + ['a', 'b', 'c']
array_2 = np.array(complex_list[:3])
print('complex_list[:3]', array_2.dtype)    # 整型
array_2 = np.array(complex_list[:6])
print('complex_list[:6]', array_2.dtype)    # 浮点型占优
array_2 = np.array(complex_list[:])
print('complex_list[:]', array_2.dtype)     # （混合数据）自定义类型
print('--------------分割线--------------')

# 使用NumPy函数生成数组
print(np.arange(9).reshape(3, 3))
print(np.arange(9)[::-1])   # 颠倒数据
print(np.random.randint(low=1, high=10, size=(3, 3)).reshape(3, 3))
print(np.linspace(start=0, stop=1, num=10))                 # 等差数列
print(np.logspace(start=0, stop=1, num=10, base=10.0))      # 等比数列
print(np.random.normal(size=(3, 3)))                        # 标准正态分布
print(np.random.normal(loc=1.0, scale=3.0, size=(3, 3)))    # 不同参数值的标准正态分布。loc：均值，scale：标准差
print(np.random.uniform(low=0.0, high=1.0, size=(3, 3)))    # 均匀分布
print('--------------分割线--------------')

# 直接从文件中获得数组
iris = np.loadtxt('iris.csv', delimiter=',', dtype=object)
print(iris[:5])
print('--------------分割线--------------')

# 从pandas中提取数据
iris_filename = 'iris.csv'
iris = pd.read_csv(iris_filename, header=None)
iris_array = iris.values
print(iris_array[:5])
print(iris_array.dtype)

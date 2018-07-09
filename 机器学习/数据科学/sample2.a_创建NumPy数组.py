import numpy as np

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
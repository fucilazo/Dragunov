"""
求 y=x^2 在[0,2]区间的积分
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 1000)     # 0~2创建1000个等差数列
y = x ** 2
plt.plot(x, y)
plt.fill_between(x, y, where=(y > 0), color='red', alpha=0.5)   # 区域填充

# 该红色区域在一个2×4的正方形里面。使用蒙特卡洛方法，随机在这个正方形里面产生大量随机点（数量为N）
# 计算有多少点（数量为count）落在红色区域内（判断条件为y<x2），count/N就是所要求的积分值，也即红色区域的面积。
N = 1000
points = [[xy[0] * 2, xy[1] * 4] for xy in np.random.rand(N, 2)]
plt.scatter([x[0] for x in points], [x[1] for x in points], s=5, c=np.random.rand(N), alpha=0.5)
plt.show()

# 计算落在红色区域的比重
count = 0
for xy in points:
    if xy[1] < xy[0] ** 2:
        count += 1
print((count / N) * (2 * 4))

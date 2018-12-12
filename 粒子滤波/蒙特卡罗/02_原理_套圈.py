"""
设物品中心点坐标为（0,0），物品半径为5cm。计算命中目标的概率
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
circle_target = mpatches.Circle([0, 0], radius=5, edgecolor='r', fill=False)
plt.xlim(-80, 80)
plt.ylim(-80, 80)
plt.axes().add_patch(circle_target)
"""
设投圈半径8cm，投圈中心点围绕物品中心点呈二维正态分布，均值μ=0cm，标准差σ=20cm，模拟1000次投圈过程。
"""
N = 1000
u, sigma = 0, 20
points = sigma * np.random.randn(N, 2) + u

plt.scatter([x[0] for x in points], [x[1] for x in points], c=np.random.rand(N), alpha=0.5)
plt.show()
# 计算1000次投圈过程中，投圈套住物品的占比情况。
print(len([xy for xy in points if xy[0] ** 2 + xy[1] ** 2 < (8-5) ** 2]) / N)

"""
figure(num=None, figsize=None, dpi=None, facecolor=None, edgecolor=None, frameon=True)
num:        图像编号或名称，数字为编号 ，字符串为名称
figsize:    指定figure的宽和高，单位为英寸；
dpi:        参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
facecolor:  背景颜色
edgecolor:  边框颜色
frameon:    是否显示边框

subplot(nrows, ncols, index, **kwargs)
nrows:行数
ncols:列数
index:序号
"""
import matplotlib.pyplot as plt
import numpy as np

# add_subplot新增子图
x = np.arange(0, 100)
fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, x)
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, x**2)
ax3.grid(color='r', linestyle='--', linewidth=1, alpha=0.3)
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, np.log(x))

plt.show()
# ---------------------------------------------------------
# add_axes新增子区域
fig = plt.figure()
x = [1, 2, 3, 4, 5, 6, 7]
y = [1, 3, 4, 2, 5, 8, 6]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8    # 从figure 10%的位置开始绘制，宽高为figure的 80%
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'r')
ax1.set_title('area1')

left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax1 = fig.add_axes([left, bottom, width, height])
ax1.plot(x, y, 'b')
ax1.set_title('area2')

plt.show()





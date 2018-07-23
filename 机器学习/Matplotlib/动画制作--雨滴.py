import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

fig = plt.figure(figsize=(6, 6), facecolor='white')     # 创建空白画板
ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)    # 建立子图
# 创建一些环形
n = 50
size_min = 50
size_max = 50*50
# 环形的坐标
P = np.random.uniform(0, 1, (n, 2))
# 环形的颜色
C = np.ones((n, 4))*(0, 0, 0, 1)
# alpha通道从0（透明）to 1（不透明）
C[:, 3] = np.linspace(0, 1, n)
# 环形的大小
S = np.linspace(size_min, size_max, n)
# 散点作图
scat = ax.scatter(P[:, 0], P[:, 1], s=S, lw=0.5, edgecolors=C, facecolors='None')
# 确定坐标轴为(0,1)，去除坐标刻度
ax.set_xlim(0, 1), ax.set_xticks([])
ax.set_ylim(0, 1), ax.set_yticks([])


# 更新函数
def update(frame):
    global P, C, S
    # 每一个环都变的更透明
    C[:, 3] = np.maximum(0, C[:, 3]-1.0/n)
    # 每一个环都变得更大
    S += (size_max - size_min)/n
    # 重新设置环形
    i = frame % 50
    P[i] = np.random.uniform(0, 1, 2)
    S[i] = size_min
    C[i, 3] = 1
    # 更新散点对象
    scat.set_edgecolors(C)
    scat.set_sizes(S)
    scat.set_offsets(P)
    # 返回更新后的对象
    return scat,    # ATTENTION!!!!


ani = animation.FuncAnimation(fig, update, interval=10, blit=True, frames=200)
plt.show()


# --------------------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# x = np.linspace(0,2*np.pi,100)
#
# fig = plt.figure()
# sub = fig.add_subplot(111, xlim=(x[0], x[-1]), ylim=(-1, 1))
# PLOT, = sub.plot([],[])
#
# def animate(i):
#     PLOT.set_data(x[:i], np.sin(x[:i]))
#     # print("test")
#     return PLOT,
#
# ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=10, blit=True)
# plt.show()
# --------------------------------------------------------------------------------------------

import numpy as np
import math
import matplotlib.pyplot as plt

x0 = 0      # 初始值
T = 500     # 仿真步数
N = 80      # 粒子滤波中粒子数，越大效果越好，计算量也越大
R = 1       # 测量噪声的协方差，且其均值为0

# 初始化高斯分布初始粒子
# v = 2   # 初始分布方差
next_particle = np.zeros(shape=(N, 1))
z_update = np.zeros(shape=(N, 1))
particle_w = np.zeros(shape=(N, 1))
current_particle = np.zeros(shape=(N, 1))
# 高斯分布产生初始粒子，就是在x状态附近做一个随机样本抽样的过程
for i in range(N):
    current_particle[i] = np.random.randn()*0.5

x_out = [x0]     # 用于存储系统状态方程计算得到的每一步的状态值
z_out = [x0]    # 用于存储量测方程计算得到的每一步的状态值
x_est = x0       # time by time output of the particle filters estimate
x_est_out = [x_est]     # 用于记录粒子滤波每一步估计的粒子均值（该均值即为对对应步状态的估计值）
x_s = [0]    # 用以存储方差

for t in range(1, T):
    if t < T/2:
        # 从状态方程当中获取下一时刻的状态值（称为预测）
        x = np.random.randn() * 0.5
        # 在当前状态下，通过观测方程获取的观测量的值
        z = x**2/2 + np.random.randn()*0.5
        for i in range(1, N):
            next_particle[i] = np.random.randn()*0.5
            z_update[i] = next_particle[i]**2/2 + np.random.randn()*0.5
            particle_w[i] = (1 / np.sqrt(2 * np.pi * R)) * np.exp(-(z - z_update[i]) ** 2 / (2 * R))    # 高斯函数
    else:
        # 从状态方程当中获取下一时刻的状态值（称为预测）
        # x = np.random.randn() * 0.5 + 5 * np.random.rand()
        x = np.random.randn() * 0.5 + 5 * np.random.rand() * np.random.randn()
        # 在当前状态下，通过观测方程获取的观测量的值
        z = x ** 2 / 2 + np.random.randn() * 0.5
        for i in range(1, N):
            next_particle[i] = np.random.randn() * 0.5
            z_update[i] = next_particle[i] ** 2 / 2 + np.random.randn() * 0.5
            # %对每个粒子计算其权重，这里假设量测噪声是高斯分布。所以 w = p(y|x)对应下面的计算公式
            particle_w[i] = (1 / np.sqrt(2 * np.pi * R)) * np.exp(-(z - z_update[i]) ** 2 / (2 * R))  # 高斯函数
    particle_w[0] = [2.21646841e-76]
    particle_w = np.divide(particle_w, np.sum(particle_w))  # 归一化

    mar = np.zeros(shape=(1, N))
    # 重采样
    for i in range(1, N):
        # 用rand函数来产生在0到1之间服从均匀分布的随机数，用于找出归一化后权值较大的粒子
        # 在这里归一化后的权值太小了，很难单个粒子的权值会大于u=rand产生的随机数，这里用累加的方式来获得具有较大权值的粒子
        mar = ((np.random.rand() <= np.cumsum(particle_w)))
        for io in range(1, N):
            # 如果大于等于，则将该权值的粒子保留下来
            if mar[io] == True:
                current_particle[i] = next_particle[io]
                # 如果找到这样的大权值粒子，则退出，寻找下一个粒子，别忘了，在每一次寻找粒子的时候，都是从头到尾的
                # 故可能会保留到重复的粒子, 所以在这里容易造成粒子样本枯竭，即粒子的多样性失去，只剩下一个大的粒子，在不断的复制自己
                break

    # 状态估计。若重采样后，每个粒子的权重都变成了1/n
    x_est = np.mean(current_particle)
    # Save data in arrays for later plotting

    x_out.append(x)
    z_out.append(z)
    x_est_out.append(x_est)
    x_s.append(np.sqrt(np.mean((x_est - x) ** 2)))

x1 = np.arange(0, T)
plt.subplot(2, 1, 1)
plt.plot(x1, x_out, label="True")
plt.plot(x1, x_est_out, label="Est")   # plotting out the tracking
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x1, x_s)
plt.show()














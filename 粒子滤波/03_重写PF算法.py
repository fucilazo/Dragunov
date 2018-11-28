import numpy as np
import math
import matplotlib.pyplot as plt

input_file = 'd03.txt'
X = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(X) for X in line.split('\t')]
        X.append(data)
X = np.array(X).T

# 初始化参数
x = 0.1     # initial actual state
x_N = 1     # 系统过程噪声的协方差（由于是一维的，这里就是指方差）
x_R = 1     # 测量的协方差
T = 75      # 共进行75次
N = 10      # 粒子数，越大效果越好，计算量也越大

# 生成高斯分布初始粒子
v = 2   # 初始分布方差
x_P_update = np.zeros(shape=(10, 1))
z_update = np.zeros(shape=(10, 1))
P_w = np.zeros(shape=(10, 1))
x_p = np.zeros(shape=(10, 1))


for i in range(N):
    x_p[i] = x + np.sqrt(v) + np.random.randn()     # 高斯分布产生初始粒子
z_out = [x ** 2 / 20 + np.sqrt(x_R) * np.random.randn()]    # 实际测量值
x_out = [x]     # the actual output vector for measurement values.
x_est = x       # time by time output of the particle filters estimate
x_est_out = [x_est]
# print(x_out)# the vector of particle filter estimates.


for t in range(1, T):
    x = 0.5 * x + 25 * x / (1 + x ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.sqrt(x_N) * np.random.randn()  # state update
    z = x ** 2 / 20 + np.sqrt(x_R) * np.random.randn()
    for i in range(1, N):   # N = number of particles
        # 从先验p(x(k)|x(k-1))中采样
        x_P_update[i] = 0.5 * x_p[i] + 25 * x_p[i] / (1 + x_p[i] ** 2) + 8 * math.cos(1.2 * (t - 1)) + np.sqrt(x_N) * np.random.randn()
        # 计算采样粒子的值，为后面根据似然去计算权重做铺垫
        z_update[i] = x_P_update[i] ** 2 / 20
        # 对每个粒子计算其权重，这里假设量测噪声是高斯分布。所以 w = p(y|x)对应下面的计算公式
        P_w[i] = (1 / np.sqrt(2 * np.pi * x_R)) * np.exp(-(z - z_update[i]) ** 2 / (2 * x_R))
    P_w[0] = [2.21646841e-76]
    P_w = np.divide(P_w, np.sum(P_w))   # 归一化

    mar = np.zeros(shape=(1, 10))
    for i in range(1, N):
        mar = ((np.random.rand() <= np.cumsum(P_w)))    # 重采样
        for io in range(1, 10):
            if mar[io] == True:
                x_p[i] = x_P_update[io]
                break

    # 状态估计。若重采样后，每个粒子的权重都变成了1/n
    x_est = np.mean(x_p)
    # Save data in arrays for later plotting
    x_out.append(x)
    z_out.append(z)
    x_est_out.append(x_est)

rmse = np.sqrt(np.mean((x_out - x_est) ** 2))
print(rmse)     # root mean squared error between the estimated and actual positions or states

x1 = np.arange(0, T)
plt.plot(x1, x_est_out, label="(x_est_out) filtered observation")   # plotting out the tracking
plt.plot(x1, x_out, label="(x_out)  observation")
# plt.scatter(x1, z_out, label='权重')

plt.legend()
plt.title('Observation and Filtered Observation')
plt.show()

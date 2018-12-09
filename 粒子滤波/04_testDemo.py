import numpy as np
import math
import matplotlib.pyplot as plt

# 状态方程：x(k)=x(k-1)/2+25*x(k-1)/(1+x(k-1)^2)+8cos(1.2(k-1))+vk;vk为噪声
# 测量方程：y(k)=x(k)^2/20+nk;nk为噪声

# 初始化参数
x0 = 0.1     # initial actual state
Q = 1       # 过程噪声的协方差，且其均值为0（由于是一维的，这里就是指方差）
R = 1       # 测量噪声的协方差，且其均值为0
T = 70      # 仿真步数
N = 100     # 粒子滤波中粒子数，越大效果越好，计算量也越大

# 生成高斯分布初始粒子
v = 2   # 初始分布方差
next_particle = np.zeros(shape=(N, 1))
z_update = np.zeros(shape=(N, 1))
particle_w = np.zeros(shape=(N, 1))
current_particle = np.zeros(shape=(N, 1))

# 高斯分布产生初始粒子，就是在x状态附近做一个随机样本抽样的过程
for i in range(N):
    current_particle[i] = x0 + np.sqrt(v) + np.random.randn()
z_out = [x0 ** 2 / 20 + np.sqrt(R) * np.random.randn()]    # 用于存储量测方程计算得到的每一步的状态值
x_out = [x0]     # 用于存储系统状态方程计算得到的每一步的状态值
x_est = x0       # time by time output of the particle filters estimate
x_est_out = [x_est]     # 用于记录粒子滤波每一步估计的粒子均值（该均值即为对对应步状态的估计值）
# print(x_out)# the vector of particle filter estimates.


for t in range(1, T):
    # 从状态方程当中获取下一时刻的状态值（称为预测）
    x = 0.5 * x0 + 25 * x0 / (1 + x0 ** 2) + 8 * np.cos(1.2 * (t - 1)) + np.sqrt(Q) * np.random.randn()
    # 在当前状态下，通过观测方程获取的观测量的值
    z = x ** 2 / 20 + np.sqrt(R) * np.random.randn()
    for i in range(1, N):   # N = number of particles
        # 从先验分布（在这里当做粒子滤波中的重要性分布）p(x(k)|x(k-1))中采样
        # 利用之前生成的粒子样本集current_particle(i)带入状态方程中，记做数组next_particle(i)
        next_particle[i] = 0.5 * current_particle[i] + 25 * current_particle[i] / (1 + current_particle[i] ** 2) + 8 * math.cos(1.2 * (t - 1)) + np.sqrt(Q) * np.random.randn()
        # 计算出通过该粒子而预测出该粒子的量测值
        z_update[i] = next_particle[i] ** 2 / 20
        # 由于上面已经计算出第i个粒子，带入观测方程后的预测值，现在与真实的测量值y进行比较，越接近则权重越大，或者说差值越小权重越大
        # 这里的权重计算是关于p(y/x)的分布，即观测方程的分布，假设观测噪声满足高斯分布，那么particle_w=p(y/x)
        particle_w[i] = (1 / np.sqrt(2 * np.pi * R)) * np.exp(-(z - z_update[i]) ** 2 / (2 * R))
 x
    particle_w[0] = [2.21646841e-76]
    particle_w = np.divide(particle_w, np.sum(particle_w))   # 归一化

    mar = np.zeros(shape=(1, 10))
    # 重采样
    for i in range(1, N):
        # 用rand函数来产生在0到1之间服从均匀分布的随机数，用于找出归一化后权值较大的粒子
        # 在这里归一化后的权值太小了，很难单个粒子的权值会大于u=rand产生的随机数，这里用累加的方式来获得具有较大权值的粒子
        mar = ((np.random.rand() <= np.cumsum(particle_w)))
        for io in range(1, 10):
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

rmse = np.sqrt(np.mean((x_out - x_est) ** 2))
print(rmse)     # root mean squared error between the estimated and actual positions or states

# 误差计算
error = [abs(x_out[i] - x_est_out[i]) for i in range(len(x_out))]

x1 = np.arange(0, T)
plt.plot(x1, x_est_out, label="Est")   # plotting out the tracking
plt.plot(x1, x_out, '--', label="True")
# plt.plot(x1, z_out, label='True')

# plt.ylim(-100,100)
plt.legend()

# plt.figure()
# plt.plot(x1, error)

plt.show()

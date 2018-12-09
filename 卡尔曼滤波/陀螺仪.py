import matplotlib.pyplot as plt
import numpy as np

# 创建噪声
NUM = 300
np.random.seed(4)
Noise = np.random.randn(NUM)  # 高斯分布随机噪声
Noise_std = np.random.randn(NUM)  # 测量噪声

X = [0] * NUM
Y = [0] * NUM

# 创建数据集     Y=sin(0.2X)
for i in range(1, NUM):
    X[i] = np.sin(0.2 * i)

Noise_std_ = np.square(np.var(Noise_std))  # 求方差
Noise_ = np.square(np.var(Noise))

P = [0] * NUM  # 每次的最优偏差
K = [0] * NUM  # 卡尔曼增益

S = X + Noise_std  # 测量值

for i in range(1, NUM):
    P[i] = np.square(P[i - 1]) + 0.1 * Noise_

    K[i] = 0.1 * np.sqrt(P[i] / (Noise_std_ + P[i]))

    Y[i] = Y[i - 1] + K[i] * (S[i] - Y[i - 1])

    P[i] = np.sqrt((1 - K[i]) * P[i])
    print(P[i])

plt.plot(X, color='r', label='True')        # 数据集
plt.plot(S, color='g', label='Scaling')     # 数据测量获取值（带噪声）
plt.plot(Y, color='b', label='Est')         # 过滤后的数据
plt.legend()

plt.show()

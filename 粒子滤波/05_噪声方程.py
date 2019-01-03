import matplotlib.pyplot as plt
import numpy as np

T = 500  # 仿真步数
x_out = []
y_out = [1]
x_0 = []

# 原始数据
for t in range(0, 250):
    x_0.append(0)
for t in range(250, 270):
    x_0.append((t-249)/20)
for t in range(270, T):
    x_0.append(1)
# 白噪声
for t in range(0, 250):
    x = np.random.randn()*0.5
    x_out.append(x)
for t in range(250, 270):
    x = np.random.randn()*0.5 + (t-249)/20
    x_out.append(x)
for t in range(270, T):
    x = np.random.randn()*0.5 + 1
    x_out.append(x)
# 有色噪声
for t in range(1, T):
    y_out.append(x_out[t]**2/2 + np.random.randn()*0.5)

# # 频谱
# def ff(x):
#     xf = np.fft.fft(x)/T
#     freqs = np.linspace(0, T/2, T/2+1)
#     xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))


x1 = np.arange(0, T)
plt.figure()
plt.subplot(3, 1, 1)
plt.title("Gaussian")
plt.plot(x1, x_out)
plt.ylim(-2, 2)
plt.subplot(3, 1, 2)
plt.title("Non-Gaussian")
plt.plot(x1, y_out)
plt.ylim(-2, 2)
plt.subplot(3, 1, 3)
plt.title("Integerant")
plt.plot(x1, x_0)
plt.ylim(-1, 2)
plt.show()

# # 绘制频谱
# xf = np.fft.fft(x_out)
# xf_abs = np.fft.fftshift(abs(xf))
# axis_xf = np.linspace(-T/2, T/2-1, num=T)
# plt.plot(axis_xf,xf_abs)
# plt.axis('tight')
# plt.xlim(0,)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

T = 500
x_out = []
y_out = [1]
# 白噪声
for t in range(0, 250):
    x = np.random.randn()*0.5
    x_out.append(x)
for t in range(250, T):
    x = np.random.randn()*0.5 + 3 * np.random.rand() * np.random.randn()*0.5
    x_out.append(x)
# 有色噪声
for t in range(1, T):
    y_out.append(x_out[t]**2/2 + np.random.randn()*0.5)

x1 = np.arange(0, T)
plt.figure()
plt.subplot(2, 1, 1)
plt.title("Gaussian")
plt.plot(x1, x_out)
plt.subplot(2, 1, 2)
plt.title("Non-Gaussian")
plt.plot(x1, y_out)
plt.show()
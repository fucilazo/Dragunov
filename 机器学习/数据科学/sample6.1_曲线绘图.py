import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 50)   # 从0到5等间距，具有50个样本
y = x
y_cos = np.cos(x)
y_sin = np.sin(x)
y_log = np.log(x)

plt.figure()
plt.subplot(121)
plt.plot(x, y_cos)
plt.plot(x, y_sin)
plt.subplot(122)
plt.plot(x, y_cos, 'bo')
plt.plot(x, y_sin, 'go')
plt.plot(x+0.5, y_cos, 'r--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.show()
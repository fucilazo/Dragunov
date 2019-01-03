import matplotlib.pyplot as plt
import numpy as np
import math

T = 20
x_out = []


# def InOutQuad(t, b, c, d):
#     if t/(d/2) < 1:
#         return c/2 * t * t + b
#     return -c/2*((-t)*(t-2)-1)+b
#
#
# for t in range(0, T):
#     x_out.append(InOutQuad(t, 1, 2, 5))
# x1 = np.arange(0, T)
# plt.plot(x1, x_out)
# x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
plt.plot(x, y)
plt.show()
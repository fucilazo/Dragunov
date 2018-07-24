import numpy as np
import matplotlib.pyplot as plt

# 标准分布
x = np.random.normal(loc=0.0, scale=1.0, size=500)
y = np.random.normal(loc=3.0, scale=1.0, size=500)
plt.hist(np.column_stack((x, y)), bins=20, histtype='bar', color=['c', 'b'], stacked=True)
plt.grid()
plt.show()
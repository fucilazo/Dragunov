from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
average = np.mean(iris.data, axis=0)
std = np.std(iris.data, axis=0)
range_ = range(np.shape(iris.data)[1])

plt.subplot(121)
plt.title('Horizontal bars')
plt.barh(range_, average, color='r', xerr=std, alpha=0.4, align='center')
plt.yticks(range_, iris.feature_names)
plt.subplot(122)
plt.title('Vertical bars')
plt.bar(range_, average, color='b', yerr=std, alpha=0.4, align='center')
plt.xticks(range_, range_)
plt.show()
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt

# 人脸识别数据集
datasets = fetch_olivetti_faces(shuffle=True, random_state=5)
for k in range(1, 7):
    plt.subplot(2, 3, k)
    plt.imshow(datasets.data[k].reshape(64, 64), cmap=plt.cm.gray, interpolation='nearest')
    plt.title('subject' + str(datasets.target[k]))
    plt.axis('off')
plt.show()

# 手写数据集
digits = load_digits()
for number in range(1, 10):
    plt.subplot(3, 3, number)
    plt.imshow(digits.images[number], cmap='binary', interpolation='none', extent=[0, 8, 0, 8])
    plt.grid()
plt.show()


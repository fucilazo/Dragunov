import numpy as np
import matplotlib.pyplot as plt


input_file = 'd03.txt'
X = []
y = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(X) for X in line.split('\t')]
        X.append(data)

X = np.array(X).T

print(X[0])

# def plot_classifier(X, y):
#     x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
#     y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
#     step_size = 0.01
#     x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
#     plt.figure()
#     plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
#     plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)
#     plt.xlim(x_values.min(), x_values.max())
#     plt.ylim(y_values.min(), y_values.max())
#     plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
#     plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
#     plt.show()
#
#
# plot_classifier(X, y)
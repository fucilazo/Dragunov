import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
"""
学习曲线可以帮助我们理解训练数据集的大小对机器学习模型的影响。当遇到计算能力限制时，这一点非常有用
"""
input_file = 'car.data.txt'
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 学习曲线
classifier = RandomForestClassifier(random_state=7)
# 分别用200，500，800，1100的训练数据集的大小测试模型的性能指标。把cv参数设置为5，就是用五折交叉验证
parameter_grid = np.array([200, 500, 800, 1100])
train_sizes, train_scores, validation_scores = learning_curve(classifier, X, y, train_sizes=parameter_grid, cv=5)
print('##### LEARNING CURVES #####')
print('Training scores:\n', train_scores)
print('Validation scores:\n', validation_scores)

# 画出学习曲线图
plt.figure()
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title('Learning curve')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.show()
"""
结果显示，虽然训练数据集的规模越小，仿佛训练准确性越高，但是它们很容易导致过度拟合
如果选择较大规模的训练数据集，就会消耗更多的资源
因此，训练规模的选择也是一个需要结合计算能力进行综合考虑的问题
"""
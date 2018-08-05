import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import validation_curve
from sklearn.ensemble import RandomForestClassifier
"""
前面用随机森林建立了分类器，但是并不知道如何定义参数。n_estimators和max_depth参数。它们被成为超参数，分类器的性能由它们来决定
验证曲线可以帮助理解每个超参数对训练得分的影响
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

# 验证曲线
classifier = RandomForestClassifier(max_depth=4, random_state=7)
parameter_grid = np.linspace(25, 200, 8).astype(int)    # 搜索空间，评估器数量会在25~200之间每隔8个数迭代一次
train_scores, validation_scores = validation_curve(classifier, X, y, 'n_estimators', parameter_grid, cv=5)
print('##### VALIDATION CURVES #####')
print('Param: n_estimators\nTraining scores:\n', train_scores)
print('Param: n_estimators\nValidation scores:\n', validation_scores)

# 画出曲线图
plt.figure()
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Number of estimators')
plt.ylabel('Accuracy')
plt.show()

# 同样，把n_estimators的值固定为20，看看max_depth参数变化对性能的影响
classifier = RandomForestClassifier(n_estimators=20, random_state=7)
parameter_grid = np.linspace(2, 10, 5).astype(int)
train_scores, valid_scores = validation_curve(classifier, X, y, 'max_depth', parameter_grid, cv=5)
print('Param: max_depth\nTraining scores:\n', train_scores)
print('Param: max_depth\nValidation scores:\n', valid_scores)
plt.figure()
plt.plot(parameter_grid, 100 * np.average(train_scores, axis=1), color='black')
plt.title('Training curve')
plt.xlabel('Max depth of the tree')
plt.ylabel('Accuracy')
plt.show()
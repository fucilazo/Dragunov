import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

input_file = 'car.data.txt'
X = []
count = 0
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# 将字符串转化为数值
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 训练分类器
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}   # 可以通过改变n_estimators'，'max_depth'的值来观察如何改变分类器的准确性
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)

# 交叉验证
accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print('Accuracy of the classifier: ', str(round(100 * accuracy.mean(), 2)), '%')

# 对单一数据进行分类
input_data = ['vhigh', 'vhigh', '2', '2', 'small', 'low']
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    input_data_encoded[i] = int(label_encoder[i].transform([input_data[i]]))
input_data_encoded = np.array(input_data_encoded)

# 预测
output_class = classifier.predict(input_data_encoded.reshape(1, -1))
print('Output class: ', label_encoder[-1].inverse_transform(output_class)[0])
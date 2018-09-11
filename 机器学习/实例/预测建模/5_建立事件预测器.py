import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

"""
building_event_binary.txt
[星期][日期][时间][离开大楼人数][进入大楼人数][是否有活动]
building_event_multiclass.txt
[星期][日期][时间][离开大楼人数][进入大楼人数][活动类型]
"""
input_file = 'building_event_binary.txt'    # 可考虑使用不同的数据集
X = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# 字符串转换成数值
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 建立SVM模型
params = {'kernel': 'rbf', 'probability': True, 'class_weight': 'balanced'}
classifier = SVC(**params)
classifier.fit(X, y)

# 交叉验证
accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=3)
print('Accuracy of the classifier: ', str(round(100 * accuracy.mean(), 2)), '%')

# 用单一数据示例进行测试
input_data = ['Tuesday', '08/26/05', '12:30:00', '21', '23']
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count += 1
input_data_encoded = np.array(input_data_encoded)

# 预测结果
output_class = classifier.predict(input_data_encoded.reshape(1, -1))
print('Output class: ', label_encoder[-1].inverse_transform(output_class)[0])
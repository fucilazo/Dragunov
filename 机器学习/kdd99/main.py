import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

input_file = 'kddcup.data_10_percent_corrected'
output_model_file = 'saved_model.pkl'   # 分类模型
X = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = line[:-1].split(',')
        X.append(data)
X = np.array(X)

# 字符串转换为数字
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

# 训练分类器
params = {'n_estimators': 200, 'max_depth': 8, 'random_state': 7}   # 可以通过改变n_estimators'，'max_depth'的值来观察如何改变分类器的准确性
classifier = RandomForestClassifier(**params)
classifier.fit(X, y)


# 存储模型文件
def saving():
    with open(output_model_file, 'wb') as f_save:
        pickle.dump(classifier, f_save)


# 使用模型文件
def reading():
    with open(output_model_file, 'rb') as f_load:
        model_classifier = pickle.load(f_load)
    return model_classifier


# # 交叉验证
# accuracy = cross_val_score(reading(), X, y, scoring='accuracy', cv=3)
# print('Accuracy of the classifier: ', str(round(100 * accuracy.mean(), 2)), '%')

# 对单一数据示例进行编码测试
input_data = ['0', 'udp', 'private', 'SF', '105', '146', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',
              '0', '0', '0', '0', '1', '1', '0.00', '0.00', '0.00', '0.00', '1.00', '0.00', '0.00', '255', '254',
              '1.00', '0.01', '0.00', '0.00', '0.00', '0.00', '0.00', '0.00']
count = 0
input_data_encoded = [-1] * len(input_data)
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]]))
        count += 1
input_data_encoded = np.array(input_data_encoded)

# 预测分类
output_class = classifier.predict(input_data_encoded.reshape(1, -1))
print(label_encoder[-1].inverse_transform(output_class)[0])

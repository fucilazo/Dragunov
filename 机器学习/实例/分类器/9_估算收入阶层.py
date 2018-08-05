import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
"""
根据[14]个属性建立分类器评估一个人的收入等级。可能的输入类型是“高于50K”和“低于或等于50K”
这个数据集有些复杂，每个数据点都是数字和字符串的混合体。数值数据是有价值的，在这种情况下不能用标记编码器进行编码
需要设计一套既可以处理数值数据，也可以处理非数值数据的系统
数据集将采用美国人口普查收入数据集中的数据
"""
input_file = 'adult.data.txt'
X = []
y = []
count_lessthan50k = 0
count_morethan50k = 0
num_images_threshold = 10000
# 将使用数据集中20000个数据点，每种数据10000个，保证初始类型没有偏差。如果大部分数据点都属于一个类型，那么分类器就会倾向于这个类型
with open(input_file, 'r') as f:
    for line in f.readlines():
        if '?' in line:     # 缺失数据
            continue

        data = line[:-1].split(', ')     # 最后一个字符为换行符

        if data[-1] == '<=50K' and count_lessthan50k < num_images_threshold:
            X.append(data)
            count_lessthan50k += 1
        elif data[-1] == '>50K' and count_morethan50k < num_images_threshold:
            X.append(data)
            count_morethan50k += 1

        if count_lessthan50k >= num_images_threshold and count_morethan50k >= num_images_threshold:
            break
X = np.array(X)

# # 将字符串转换为数值数据
label_encoder = []
X_encoded = np.empty(X.shape)
# 由单行数据推断元素类型（数字/字符串）
for i, item in enumerate(X[0]):     # i：序号；item：元素
    if item.isdigit():              # 数据为数值则保持不变
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# 建立分类器
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
# 提取性能指标，计算分类器的F1得分
f1 = cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=5)
print('F1 score: ', str(round(100 * f1.mean(), 2)), '%')

# 交叉验证
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
classifier_gaussiannb_test = GaussianNB()
classifier_gaussiannb_test.fit(X_train, y_train)
y_test_pred = classifier_gaussiannb_test.predict(X_test)
# 测试数据准确度
accuracy_formal = 100 * (y_test == y_test_pred).sum() / X_test.shape[0]
print('Accuracy: ', round(accuracy_formal, 2), '%')
accuracy = cross_val_score(classifier_gaussiannb_test, X_test, y_test, scoring='precision_weighted', cv=5)
print('Precision: ', str(round(100 * accuracy.mean(), 2)), '%')

# 对单一数据示例进行编码测试
input_data = ['39', 'State-gov', '77516', 'Bachelors', '13', 'Never-married', 'Adm-clerical', 'Not-in-family', 'White',
              'Male', '2174', '0', '40', 'United-States']
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
output_class = classifier_gaussiannb.predict(input_data_encoded.reshape(1, 14))
print(label_encoder[-1].inverse_transform(output_class))

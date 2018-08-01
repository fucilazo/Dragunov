import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = np.array([[3, -1.5, 2,  -5.4],   # 样本数据
                [0,  4,  -0.3,  2.1],
                [1, 3.3, -1.9, -4.3]])

# 标准化处理--把每个特征的平均值移除，以保证特征均值无限接近0。这样做可以消除特征彼此间的偏差--->每列三个元素相加接近0
data_standardized = preprocessing.scale(data)
print('标准化处理：\n', data_standardized, '\n', '-'*50)
print('均值：\n', data_standardized.mean(axis=0), '\n', '-'*50)    # 均值几乎为0
print('标准差：\n', data_standardized.std(axis=0), '\n', '-'*50)    # 标准差为1

# 范围缩放
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # 指定缩放范围
data_scaled = data_scaler.fit_transform(data)   # 拟合数据并转化为标准形式
print('缩放后的数据：\n', data_scaled, '\n', '-'*50)

# 归一化
data_normalized = preprocessing.normalize(data, norm='l1')  # 特征向量调整为L1范数，使特征向量的数值之和为1
print('L1 normalized data：\n', data_normalized, '\n', '-'*50)

# 二值化
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)     # 将数值特征向量转换为布尔类型向量
print('Binarized data：\n', data_binarized, '\n', '-'*50)

# 独热编码  可用来解决文本型数据的分类
"""
========
0 2 1 12    第一列：[0, 1, 2, 1]有三种值：0,1,2  -->{0:100, 1:010, 2:001}
1 3 5 3     第二列：[2, 3, 3, 2]有两种值：2,3    -->{2:10, 3:01}
2 3 2 12    第三列：[1, 5, 2, 4]有四种值：1,2,4,5-->{1:1000, 2:0100, 4:0010, 5:0001}
1 2 4 3     第四列：[12,3, 12,3]有两种值：3,12   -->{3:10, 12:01}
========
[2, 3, 5, 3]===>[001, 01, 0001, 10]-->00101000110
"""
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoder_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print(encoder_vector)



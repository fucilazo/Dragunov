import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from sklearn import metrics

file_name = 'data_singlevar.txt'
X = []
y = []
with open(file_name, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)


num_training = int(0.8 * len(X))
num_test = len(X) - num_training
X_train = np.array(X[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])
X_test = np.array(X[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

output_model_file = 'saved_model.pkl'


# 存储模型文件
def saving():
    with open(output_model_file, 'wb') as f:
        pickle.dump(linear_regressor, f)


# 使用模型文件
def reding():
    with open(output_model_file, 'rb') as f:
        model_linregr = pickle.load(f)
    return model_linregr


if __name__ == '__main__':
    y_test_pred_new = reding().predict(X_test)
    print(metrics.mean_absolute_error(y_test, y_test_pred_new))
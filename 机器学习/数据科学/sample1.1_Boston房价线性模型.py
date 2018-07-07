import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# 波士顿房价数据集
bostom_dataset = datasets.load_boston()

X_full = bostom_dataset.data
Y = bostom_dataset.target
print(X_full.shape)
print(Y.shape)

selector = SelectKBest(f_regression, k=1)   # 选择最具有判别能力的SelectKBest类别作为特征
selector.fit(X_full, Y)                     # 采用.fit()方法进行数据拟合
X = X_full[:, selector.get_support()]       # 通过.get_support()方法将数据集缩减成一个向量
print(X.shape)

# 数据集坐标系
# plt.scatter(X, Y, color='black')
# plt.show()

# regressor = LinearRegression(normalize=True)  # 假定X、Y存在形如Y=a+bX的线性关系，估计模型的参数a、b
# regressor = SVR()                             # SVM线性回归模型
regressor = RandomForestRegressor()
regressor.fit(X, Y)
plt.scatter(X, Y, color='black')
# plt.plot(X, regressor.predict(X), color='blue', linewidth=3)  # 每个点直线相连
plt.scatter(X, regressor.predict(X), color='blue', linewidth=3)
plt.show()
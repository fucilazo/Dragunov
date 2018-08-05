import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
"""
精度（prcision）是指被分类器正确分类的样本数量占分类器总分类样本数量的百分比
召回率（recall）是指本应正确分类的样本数量占某分类总样本数量的百分比
F1得分指标是精度和召回率的合成指标，实际上是精度和召回率的调和平均值：F1 = 2 x 精度 x 召回率/(精度+召回率)
"""
input_file = 'data_multivar.txt'
X = []
y = []

with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(X) for X in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

# 建立一个朴素贝叶斯分类器
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

num_validations = 5
accuracy = cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print('精度= ', str(round(100 * accuracy.mean(), 2)), '%')

f1 = cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
print('F1: ', str(round(100 * f1.mean(), 2)), '%')

precision = cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
print('Precision: ', str(round(100 * precision.mean(), 2)), '%')

recall = cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validations)
print('Recall', str(round(100 * recall.mean(), 2)), '%')
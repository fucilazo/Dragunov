import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
"""
学习曲线作用在不同的样本规模上，验证曲线则表示训练和交叉验证性能与超参数数值之间的关系
"""
testing_range = np.logspace(-5, 2, 8)
digits = load_digits()
x, y = digits.data, digits.target

hypothesis = SGDClassifier(loss='log', shuffle=True, max_iter=5, penalty='l2', alpha=0.0001, random_state=3)
train_scores, test_scores = validation_curve(hypothesis, x, y, param_name='alpha', param_range=testing_range,
                                             cv=10, scoring='accuracy')
mean_train = np.mean(train_scores, axis=1)
upper_train = np.clip(mean_train + np.std(train_scores, axis=1), 0, 1)
lower_train = np.clip(mean_train - np.std(train_scores, axis=1), 0, 1)
mean_test = np.mean(test_scores, axis=1)
upper_test = np.clip(mean_test + np.std(test_scores, axis=1), 0, 1)
lower_test = np.clip(mean_test - np.std(test_scores, axis=1), 0, 1)
plt.semilogx(testing_range, mean_train, 'ro-', label='Training')
plt.fill_between(testing_range, upper_train, lower_train, alpha=0.1, color='r')
plt.semilogx(testing_range, mean_test, 'bo-', label='Cross-validation')
plt.fill_between(testing_range, upper_test, lower_test, alpha=0.1, color='b')
plt.grid()
plt.xlabel('alpha parameter')
plt.ylabel('accuracy')
plt.ylim(0.8, 1.0)
plt.legend(loc='lower left', numpoints=1)
plt.show()
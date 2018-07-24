import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
"""
就训练效果而言，对结果的期待开始时应该高，然后会下降。然而，根据假设的偏差和方差水平不同，就会发现有不同的结果：
    1.高偏差的机器学习算法倾向于从平均性能开始，当遇到更多复杂数据时性能迅速降低，然后，无论增加多少实例都保持在相同的水平。
      低偏差的机器学习算法在样本多时能够更好地泛化，但是只适用于相似的复杂数据结构，因此也限制了算法的性能
    2.高方差的假设往往开始时性能很好，然后随着增加更多的实例，性能会慢慢降低。原因是它记录了大量训练样本特征
至于交叉验证，我们注意到两个表现：
    1.高偏差的假设往往从低性能开始，但它的增长速度非常迅速，直到达到几乎与训练数据相同的性能。然后，它的性能不再提升
    2.高方差的假设往往从非常低的性能开始，然后平稳又缓慢地提高性能，这是因为更多的实例有助于提高泛化能力。
      它很难达到训练集上的性能，在它们之间总有一段差距
"""
digits = load_digits()
x, y = digits.data, digits.target
# 梯度下降假设检验
hypothesis = SGDClassifier(loss='log', shuffle=True, max_iter=5, penalty='l2', alpha=0.0001, random_state=3)
# 使统计数据可视化以判断是否表现为高偏差或高方差
train_size, train_scores, test_scores = learning_curve(hypothesis, x, y, train_sizes=np.linspace(0.1, 1.0, 5), cv=10,
                                                       scoring='accuracy', exploit_incremental_learning=False)
mean_train = np.mean(train_scores, axis=1)
upper_train = np.clip(mean_train + np.std(train_scores, axis=1), 0, 1)
lower_train = np.clip(mean_train - np.std(train_scores, axis=1), 0, 1)
mean_test = np.mean(test_scores, axis=1)
upper_test = np.clip(mean_test + np.std(test_scores, axis=1), 0, 1)
lower_test = np.clip(mean_test - np.std(test_scores, axis=1), 0, 1)
plt.plot(train_size, mean_train, 'ro-', label='Training')
plt.fill_between(train_size, upper_train, lower_train, alpha=0.1, color='r')
plt.plot(train_size, mean_test, 'bo-', label='Cross-validation')
plt.fill_between(train_size, upper_test, lower_test, alpha=0.1, color='b')
plt.grid()
plt.xlabel('sample size')
plt.ylabel('accuracy')
plt.legend(loc='lower right', numpoints=1)
plt.show()
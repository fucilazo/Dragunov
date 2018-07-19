"""
逻辑回归通过拟合曲线（或者学习超平面）实现分类，决策树通过寻找最佳划分特征进而学习样本路径实现分类，支持向量机通过寻找分类超平面进而最大化类别间隔实现分类。相比之下，朴素贝叶斯独辟蹊径，通过考虑特征概率来预测分类。
朴素贝叶斯把类似“笑”这样的特征概率化，构成一个“人的样貌向量”以及对应的“好人/坏人标签”，训练出一个标准的“好人模型”和“坏人模型”，这些模型都是各个样貌特征概率构成的。
这样，当一个品行未知的人来以后，我们迅速获取ta的样貌特征向量，分布输入“好人模型”和“坏人模型”，得到两个概率值。如果“坏人模型”输出的概率值大一些，那这个人很有可能就是个大坏蛋了。

决策树是怎么办的呢？决策树可能先看性别，因为它发现给定的带标签人群里面男的坏蛋特别多，这个特征眼下最能区分坏蛋和好人，然后按性别把一拨人分成两拨；接着看“笑”这个特征，因为它是接下来最有区分度的特征，然后把两拨人分成四拨；接下来看纹身，，，，
最后发现好人要么在田里种地，要么在山上砍柴，要么在学堂读书。而坏人呢，要么在大街上溜达，要么在地下买卖白粉，要么在海里当海盗。

向量化、矩阵化操作是机器学习的追求。从数学表达式上看，向量化、矩阵化表示更加简洁；在实际操作中，矩阵化（向量是特殊的矩阵）更高效。
"""
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))


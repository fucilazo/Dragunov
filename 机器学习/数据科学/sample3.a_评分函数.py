import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# 多标号分类
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=4)
classifier = DecisionTreeClassifier(max_depth=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(iris.target_names)
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)   # 第0类从来没被误分为其他类，第1类有三次被分为第2类，第2类有两次被误分为第1类
img = plt.matshow(cm, cmap=plt.cm.winter)
"""
cmaps = [('Perceptually Uniform Sequential',  
                            ['viridis', 'inferno', 'plasma', 'magma']),  
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',  
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',  
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',  
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),  
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',  
                             'copper', 'gist_heat', 'gray', 'hot',  
                             'pink', 'spring', 'summer', 'winter']),  
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',  
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',  
                             'seismic']),  
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',  
                             'Pastel2', 'Set1', 'Set2', 'Set3']),  
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',  
                             'brg', 'CMRmap', 'cubehelix',  
                             'gnuplot', 'gnuplot2', 'gist_ncar',  
                             'nipy_spectral', 'jet', 'rainbow',  
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]  
"""
plt.colorbar(img, ticks=range(0, 31, 3))    # 理想情况下，一个完美的分类，其混淆矩阵所有的“非”对角线上的元素都应为0
plt.show()
print('Accuracy:', metrics.accuracy_score(y_test, y_pred))  # 准确率。分类正确率
# print('Precision:', metrics.precision_score(y_test, y_pred))    # 精确度。计算每一个分类标号集合中正确分类的数量，然后对所有标号的结果进行平均
# print('Recall:', metrics.recall_score(y_test, y_pred))   # 召回率。正确分类标号的数量除以该类标号的总数，然后结果取平均
# print('F1 score:', metrics.f1_score(y_test, y_pred))    # F1分值。它是精确度和召回率的调和平均
print(classification_report(y_test, y_pred, target_names=iris.target_names))
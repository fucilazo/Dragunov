"""
在监督学习种经常要处理各种各样的标记，这些标记可能使文本，也可能是数字。如果是数字，那么算法可以直接使用它们
但是在许多情况下，标记需要以人们可以理解的形式存在。标记编码就是要把文本标记转换成数值形式，来让算法懂得如何操作标记
"""
from sklearn import preprocessing

# 定义编码
label_encoder = preprocessing.LabelEncoder()    # 定义一个标记编码器
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']   # 创建一些标记
label_encoder.fit(input_classes)
print('标记编码：')
for i, item in enumerate(label_encoder.classes_):
    print('%s --> %d' % (item, i))
print('-'*50)

# 转换编码
labels = ['toyota', 'ford', 'audi']
encoder_labels = label_encoder.transform(labels)    # 转换成之前的编码
print('标记=', labels, '\n编码=', encoder_labels, '\n', '-'*50)

# 检查编码
encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('编码=', encoded_labels, '\n标记=', list(decoded_labels))
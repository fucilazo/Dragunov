from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)
print(twenty_sci_news.data[0], '\n')
print(twenty_sci_news.filenames, '\n')
print(twenty_sci_news.target[0], '\n')    # 分类数据：0->sci.med主题 1->sci.space主题

count_vect = CountVectorizer()                                  # 初始化一个CountVectorizer对象
word_count = count_vect.fit_transform(twenty_sci_news.data)     # 调用算法计算每个文档中单词的数量
print(word_count.shape, '\n')     # 输出为稀疏矩阵，(观测样本数(文档数)，特征数(数据集中的单词数))
# 经过CountVectorizer转换，每篇文档都与其特征向量相关
print(word_count[0], '\n')    # 输出是一个只有非零元素的稀疏矢量

# 单词出现次数的直接对应
word_list = count_vect.get_feature_names()
for n in word_count[0].indices:
    print('word:', word_list[n], 'appears', word_count[0, n], 'times')

# 计算单词出现频率，总和约为'1'
tf_vect = TfidfVectorizer(use_idf=False, norm='l1')     # 选择l2范数时可以增大罕见词和常见词的差别
word_freq = tf_vect.fit_transform(twenty_sci_news.data)
word_list = tf_vect.get_feature_names()
for n in word_freq[0].indices:
    print('word:', word_list[n], 'has frequency', word_freq[0, n])

# 以增大罕见词与常见词频率差距的更有效计算方法是使用Tfidf对文本数据进行向量化
tfidf_vect = TfidfVectorizer()
word_tfidf = tfidf_vect.fit_transform(twenty_sci_news.data)
word_list = tfidf_vect.get_feature_names()
for n in word_tfidf[0].indices:
    print('Word:', word_list[n], 'has tfidf', word_tfidf[0, n])
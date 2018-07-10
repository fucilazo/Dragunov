import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)
tf_vect = TfidfVectorizer()
word_freq = tf_vect.fit_transform(twenty_sci_news.data)
tsvd_2c = TruncatedSVD(n_components=50)
tsvd_2c.fit(word_freq)
print(np.array(tf_vect.get_feature_names())[tsvd_2c.components_[20].argsort()[-10:][::-1]])

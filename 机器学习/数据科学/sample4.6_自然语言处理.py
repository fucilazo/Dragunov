import nltk
from nltk.stem import *

# 简单分词
my_text = "The coolest job in the next 10 years will be statisticians. People think I'm joking, but who would've " \
          "guessed that computer engineers would've been the coolest job of the 1990s?"
simple_tokens = my_text.split(' ')
print(simple_tokens)

# NLTK分词
nltk_tokens = nltk.word_tokenize(my_text)
print(nltk_tokens)

# Lancaster词干提取算法
stemmer = LancasterStemmer()
print([stemmer.stem(word) for word in nltk_tokens])     # 所有单词都是小写，且单词statisticians变为stat等等

# 词性标注
print(nltk.pos_tag(nltk_tokens))
import urllib.request
from sklearn.datasets import load_svmlight_file

target_page = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a'
a2a = urllib.request.urlopen(target_page)
X_train, Y_train = load_svmlight_file(a2a)
print(X_train.shape, Y_train.shape)


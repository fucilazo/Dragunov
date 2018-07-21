import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC


def ijcnn():
    X_train, y_train = load_svmlight_file('ijcnn1.bz2')
    first_rows = 2500   # 观测数值
    X_train, y_train = X_train[:first_rows, :], y_train[:first_rows]
    hypothesis = SVC(kernel='rbf', degree=2, random_state=101)
    scorer = cross_val_score(hypothesis, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(np.mean(scorer), np.std(scorer))


def poker():
    X_train, y_train = load_svmlight_file('poker.bz2')
    hot_encoding = OneHotEncoder(sparse=True)
    X_train = hot_encoding.fit_transform(X_train.toarray())
    hypothesis = LinearSVC(dual=False)
    scorer = cross_val_score(hypothesis, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    print(np.mean(scorer), np.std(scorer))


if __name__ == '__main__':
    ijcnn()
    poker()
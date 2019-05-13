import os
import glob
import numpy as np
import random
from skimage import io, transform
'''
从path中读取图片文件
文件存储方式：path - class1 - pic1.jpg ...
                   - class2 - picn.jpg ...
'''


def load_data(path, imgw, imgh):
    # 如果把文件夹目录下为文件下的路径加入集合
    classes = [path+x for x in os.listdir(path) if os.path.isdir(path+x)]

    imgs = []
    labels = []

    for label, folder in enumerate(classes):
        # 读取xxx/*.jpg
        for img in glob.glob(folder + '/*jpg'):
            img = io.imread(img)
            img = transform.resize(img, (imgw, imgh))
            imgs.append(img)
            labels.append(label)
    return np.asarray(imgs, dtype=np.float32), np.array(labels, dtype=np.int32)     # 记住labels一定要转换成int32形式


''' 
样本打乱顺序
'''


def disrupt_order(data, labels):
    total_num = data.shape[0]
    order_arrlist = np.arange(total_num)
    random.shuffle(order_arrlist)
    data = data[order_arrlist]
    labels = labels[order_arrlist]
    return data, labels


'''
分割测试训练集
'''


def get_train_test_data(data, labels, percent):
    seg_point = int(data.shape[0] * percent)

    x_train = data[0:seg_point, :]
    x_test = data[seg_point:, :]

    y_train = labels[0:seg_point]
    y_test = labels[seg_point:]

    return x_train, y_train, x_test, y_test


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

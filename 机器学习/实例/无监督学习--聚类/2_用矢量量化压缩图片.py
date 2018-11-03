import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn import cluster

input_file = 'test.jpg'
# 加载原图片
input_image = misc.imread(input_file, True).astype(np.uint8)


# 压缩图片
def compress_image(img, num_clusters):
    # 将图像转换成 (样本量，特征量) 数组，以运行K聚类算法
    X = img.reshape((-1, 1))
    # 对输入数据进行K聚类
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    # 为每个数据配置离它最近的中心点，并转换为图片的形状
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed


# 显示原图片
def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)


# 设置参数
def params(num_bits):   # num_bits should be between 1 and 8
    num_clusters = np.power(2, num_bits)
    compression_rate = round(100 * (8.0 - num_bits) / 8.0, 2)
    return [num_clusters, compression_rate]


plot_image(input_image, 'Original image')

num_bits = 4
print('The size of the image will be reduced by a factor of', 8.0/num_bits)
print('Compression rate = ' + str(params(num_bits)[1]) + '%')
input_image_compressed = compress_image(input_image, params(num_bits)[0])
plot_image(input_image_compressed, 'Compressed image; compression rate = ' + str(params(num_bits)[1]) + '%')

num_bits = 2
print('The size of the image will be reduced by a factor of', 8.0/num_bits)
print('Compression rate = ' + str(params(num_bits)[1]) + '%')
input_image_compressed = compress_image(input_image, params(num_bits)[0])
plot_image(input_image_compressed, 'Compressed image; compression rate = ' + str(params(num_bits)[1]) + '%')

num_bits = 1
print('The size of the image will be reduced by a factor of', 8.0/num_bits)
print('Compression rate = ' + str(params(num_bits)[1]) + '%')
input_image_compressed = compress_image(input_image, params(num_bits)[0])
plot_image(input_image_compressed, 'Compressed image; compression rate = ' + str(params(num_bits)[1]) + '%')

plt.show()
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


# 初始化权重函数
def weight_variable(shape):
    # 使用截断的正态分布来初始化权重,并保持权重的方差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个2*2的最大池化层
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 用来计算权重和偏置的均值和方差
def variable_summaries(var):
    # 统计参数的均值,并记录
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
    # 计算参数的标准差
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
    # 统计参数的最大最小值
    tf.summary.scalar("max", tf.reduce_max(var))
    tf.summary.scalar("min", tf.reduce_min(var))
    # 用直方图统计参数的分布
    tf.summary.histogram("histogram", var)


# tensorboard的日志保存目录
log_dir = "log2"
if __name__ == "__main__":
    with tf.name_scope("input"):
        # 定义输入变量
        x = tf.placeholder("float", shape=[None, 784])
        # 定义输出变量
        y_ = tf.placeholder("float", shape=[None, 10])
    with tf.name_scope("input_image"):
        # 将输入的x转成一个4D向量，第2、3维对应图片的宽高，最后一维代表图片的颜色通道数
        # 输入的图像为灰度图，所以通道数为1，如果是RGB图，通道数为3
        # tf.reshape(x,[-1,28,28,1])的意思是将x自动转换成28*28*1的数组
        # -1的意思是代表不知道x的shape，它会按照后面的设置进行转换
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # 将image保存到tensorboard中
        tf.summary.image("input", x_image, 10)
    with tf.name_scope("conv1"):
        # 初始化权重,第一层卷积，32的意思代表的是输出32个通道
        # 其实，也就是设置32个卷积，每一个卷积都会对图像进行卷积操作
        with tf.name_scope("weights"):
            w_conv1 = weight_variable([5, 5, 1, 32])
            variable_summaries(w_conv1)
        # 初始化偏置项
        with tf.name_scope("bias"):
            b_conv1 = bias_variable([32])
            variable_summaries(b_conv1)
        with tf.name_scope("relu_output"):
            # 第一层卷积
            conv1 = conv2d(x_image, w_conv1) + b_conv1
            # Relu激活函数
            h_conv1 = tf.nn.relu(conv1)
        with tf.name_scope("max_pool"):
            # 池化
            h_pool1 = max_pool_2_2(h_conv1)
    # tf.summary.image("conv1_relu_image",reverse_conv2d(conv1,[5,5,1,32],[50,28,28,1]),10)
    # #保存通过激活函数之后的图片
    # tf.summary.image("conv1_relu_image",reverse_conv2d(h_conv1,[5,5,1,32],[50,28,28,1]),10)
    # #保存经过最大池化的图片
    # tf.summary.image("conv1_max_pool_image",reverse_conv2d(h_conv1,[5,5,1,32],[50,28,28,1]),10)
    with tf.name_scope("conv2"):
        with tf.name_scope("weights"):
            # 第二层卷积
            # 初始权重
            w_conv2 = weight_variable([5, 5, 32, 64])
            variable_summaries(w_conv2)
        with tf.name_scope("bias"):
            # 初始化偏置项
            b_conv2 = bias_variable([64])
            variable_summaries(b_conv2)
        with tf.name_scope("relu_output"):
            # 第二层卷积
            conv2 = conv2d(h_pool1, w_conv2) + b_conv2
            h_conv2 = tf.nn.relu(conv2)
        with tf.name_scope("max_pool"):
            # 池化
            h_pool2 = max_pool_2_2(h_conv2)
    # 反卷积层
    with tf.name_scope("reverse_conv1") as scope:
        reverse_weight1 = weight_variable([5, 5, 32, 64])
        reverse_conv1 = tf.nn.conv2d_transpose(conv2, reverse_weight1, [50, 14, 14, 32], strides=[1, 1, 1, 1],
                                               padding="SAME")
        reverse_weight2 = weight_variable([5, 5, 1, 32])
        reverse_conv2 = tf.nn.conv2d_transpose(reverse_conv1, reverse_weight2, [50, 28, 28, 1], strides=[1, 2, 2, 1],
                                               padding="SAME")

        reverse_weight3 = weight_variable([5, 5, 1, 32])
        reverse_conv3 = tf.nn.conv2d_transpose(conv1, reverse_weight3, [50, 28, 28, 1], strides=[1, 1, 1, 1],
                                               padding="SAME")
    tf.summary.image("reverse_conv2", reverse_conv2, 10)
    tf.summary.image("reverse_conv1", reverse_conv3, 10)
    with tf.name_scope("fc1"):
        with tf.name_scope("weights"):
            # 设置全连接层的权重
            w_fc1 = weight_variable([7 * 7 * 64, 1024])
            variable_summaries(w_fc1)
        with tf.name_scope("bias"):
            # 设置全连接层的偏置
            b_fc1 = bias_variable([1024])
            variable_summaries(b_fc1)
        with tf.name_scope("relu_output"):
            # 将第二层卷积池化后的结果，转成一个7*7*64的数组
            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            # 通过全连接之后并激活
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
        with tf.name_scope("dropout"):
            # 防止过拟合
            keep_prob = tf.placeholder("float")
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    with tf.name_scope("fc2"):
        with tf.name_scope("weights"):
            # 输出层
            w_fc2 = weight_variable([1024, 10])
            variable_summaries(w_fc2)
        with tf.name_scope("bias"):
            b_fc2 = bias_variable([10])
            variable_summaries(b_fc2)
        with tf.name_scope("output"):
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # 日志输出，每迭代100次输出一次日志
    # 定义交叉熵为损失函数
    with tf.name_scope("loss"):
        cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    tf.summary.scalar("loss", cross_entropy)
    with tf.name_scope("train"):
        # 最小化交叉熵
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope("accuracy"):
        with tf.name_scope("correction_prediction"):
            # 计算准确率
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar("accuracy", accuracy)
    sess = tf.Session()

    # 合并所有的summary
    merged = tf.summary.merge_all()
    # 写到指定的磁盘路径中
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test")
    # 初始化变量
    sess.run(tf.initialize_all_variables())
    # 下载minist的手写数字的数据集
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 500 == 0:
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0},
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step%03d" % i)
            train_writer.add_summary(summary, i)

            # 记录测试集的summary
            batch_test = mnist.test.next_batch(50)
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print("test Accuracy at step %s:%s" % (i, acc))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
    print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

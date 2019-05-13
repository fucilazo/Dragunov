'''

author : Fangtao

该脚本读取path目录下的图片样本，处理后输入神经网络
神经网络来源于脚本models.py，可以多个模型中选一个
样本预处理函数保存在helpFunctions.py中
过程中保存acc和卷积后的图片等信息，可以通过tensorboard可视化

'''
from skimage import io,transform
# glob用于查找符合特定规则的路径名
import glob
import os
import tensorflow as tf
import numpy as np
import time
from helpFunctions import *
from model import *

path = 'flower_photos/'
model_saved_path = 'ckpt/model.ckpt'
w = 100
h = 100

print("loading files...")
data, labels = load_data(path, w ,h)
print('files loaded')

total_num = data.shape[0]

'''打乱样本，分散不同样本顺序'''
data , labels = disrupt_order(data , labels)

'''分割数据集'''
x_train, y_train, x_test, y_test = get_train_test_data(data, labels, 0.8)


print('train data size:',x_train.shape)
print('train label size:',y_train.shape)
#
x = tf.placeholder(dtype=tf.float32, shape=[None, w, h, 3],name='x')
y = tf.placeholder(dtype=tf.int32, shape=[None,],name='y')

regulizer = tf.contrib.layers.l2_regularizer(0.0001)

# 数据流一遍通过网络计算输出
# output = AlexNet(x, True, regulizer, 5)
output = inference(x, True, regulizer)

#(小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(output,b,name='logits_eval')

# 计算损失（多分类softmax）
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)

#
'''
使用优化器减小损失
如果acc一直上不去，很可能是lr设置太大，导致无法收敛到合理的最优解
'''
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

'''
计算正确率
先计算出每行output最大值，也就是最可能的预测label，用arg_max
为了和y比较，要cast成相同类型数据，tf.int32
然后和y比较相同数据，用equals
最后reduce_meam计算下平均对了几个,在这之前还要cast成float32类型才能计算平均值
'''
with tf.name_scope('loss'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.arg_max(output,1),dtype=tf.int32),y),tf.float32))
    tf.summary.scalar('accuracy',acc)  # tfboard一维常量可以放进summary显示图表

n_epoch = 1
batch_size=64
saver = tf.train.Saver()
sess = tf.Session()

'''tensorboard 可视化'''
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('DeskTop/',sess.graph)

''' 如果加载以前的网络 '''
saver = tf.train.import_meta_graph('ckpt/model.ckpt.meta')  # 加载网络结构
saver.restore(sess, tf.train.latest_checkpoint("ckpt/"))  # 加载检查点

''' 不加载就随机初始化'''
# sess.run(tf.global_variables_initializer())

'''
每批次都先训练，再测试
训练和测试都分成小批次
每批训练完后都保存
'''
for epoch in range(n_epoch):
    start_time = time.time()
    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_batch, y_train_batch in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([optimizer,loss,acc], feed_dict={x: x_train_batch, y: y_train_batch})  # 注意这里一定要一批一批的放！！！放多了维度不对conv2d他妈的竟然报错我日老子找了俩小时

        train_loss += err
        train_acc += ac
        n_batch += 1
    print("   train loss: %f" % (np.sum(train_loss) / n_batch))
    print("   train acc: %f" % (np.sum(train_acc) / n_batch))


    test_acc, test_loss, n_batch = 0, 0, 0
    for x_test_batch, y_test_batch in minibatches(x_test , y_test, batch_size,shuffle=False):   # 测试时不需要打乱
        err, ac = sess.run([loss, acc], feed_dict={x : x_test_batch, y : y_test_batch })
        test_loss += err
        test_acc += ac
        n_batch += 1

        '''
        这两句是绘制折线图等图表的
        把每次训练过程中的变量数值加到summary当中最后绘制
        如果训练中一个变量都不加进来的话就会报错：None
        '''
        result = sess.run(merged, feed_dict={x: x_train_batch, y: y_train_batch})  # merged也是需要run的
        writer.add_summary(result,epoch)  # result是summary类型的，需要放入writer中，i步数（x轴）

    print("   test loss: %f" % (np.sum(test_loss) / n_batch))
    print("   test acc: %f" % (np.sum(test_acc) / n_batch))
    saver.save(sess,model_saved_path)
sess.close()
'''可视化comda prompt里面调用这句：tensorboard --logdir D:\代码\BaoyanLearn\Desktop '''

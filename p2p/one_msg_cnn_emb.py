import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json

word_size = 100
embedding_size = 60
batch_size =1024
#长度大于10,正反各占一半
with open('/usr/local/oybb/project/xingqiao_data/oneMsg100') as f:
    data=json.loads(f.read())

train_data=data[:-1*batch_size]
test_data=data[-1*batch_size:]

print('data pre-processing is done')

with open('/usr/local/oybb/yzx_w2vec/30500000')as f:
    embeddings = json.loads(f.read())


def next():
    x,y=[],[]
    for i in np.random.randint(0,len(train_data),batch_size):
        data=train_data[i]
        x.append(data[:-1])
        y.append(data[-1])
    return x,y


x_test, y_test = [i[:-1] for i in test_data],[i[-1] for i in test_data]

x = tf.placeholder(shape=[batch_size, word_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

embeddings = tf.constant(embeddings)
embed = tf.nn.embedding_lookup(embeddings, x)
X = tf.reshape(embed, [batch_size, word_size, embedding_size, 1])

c1 = ml.conv2d(X, conv_filter=[3, embedding_size, 1, 2], padding='VALID', ksize=[1, 10, 1, 1], pool_stride=[1, 4, 1, 1],
               pool_padding='SAME')
c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 5, 1, 1],
               pool_padding='SAME')
# lay1 = tf.reshape(c2, [batch_size, -1])
# lay2 = ml.layer_basic(ml.bn(lay1), 1)
out=tf.reshape(c2,shape=[batch_size,20])
y = tf.nn.sigmoid(ml.layer_basic(out,1) )[:,0]
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
gv= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#loss=tf.reduce_mean((y-y_)**2)
l2_loss=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.05, scope=None), weights_list=gv)
all_loss=loss+l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................', sum(y_test) / len(y_test))

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 10 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_loss = sess.run(loss, feed_dict={x: x_test, y_: y_test})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        qtest = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size

        train_y = sess.run(y, feed_dict={x: x_train, y_: y_train})
        qtrain = 1 - len([i for i in train_y + y_train if 0.5 <= i <= 1.5]) / batch_size

        print(train_loss, test_loss, qtrain, qtest, sum(test_y * y_test) / sum(test_y),
              sum(test_y * y_test) / sum(y_test))

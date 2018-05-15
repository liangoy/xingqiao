import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json

msg_count = 1000
msg_size = 100
embedding_size = 60
batch_size = 128
col = pymongo.MongoClient().xingqiao.msgCut
data = list(col.find({'createTime': {'$lt': '2017/11/5'}}, {'_id': 0, 'status': 1, 'msgCut': 1}))

data_x = [i['msgCut'] for i in data]

for i in data_x:
    for j in range(len(i)):
        i[j].extend([0] * msg_size)
        i[j] = i[j][:msg_size]
for i in range(len(data_x)):
    data_x[i].extend([[0] * msg_size] * msg_count)
    data_x[i] = data_x[i][:msg_count]

data_y = [i['status'] for i in data]

train_data_x = data_x[:-1 * batch_size]
train_data_y = data_y[:-1 * batch_size]
test_data_x = data_x[-1 * batch_size:]
test_data_y = data_y[-1 * batch_size:]

cnt = len(train_data_y) - 2 * len([i for i in train_data_y if i == 0])
while cnt:
    r2 = np.random.randint(len(train_data_y))
    if train_data_y[r2] != 0:
        continue
    r1 = np.random.randint(len(train_data_y))
    train_data_x.insert(r1, train_data_x[r2])
    train_data_y.insert(r1, train_data_y[r2])
    cnt -= 1

cnt = len(test_data_y) - 2 * len([i for i in test_data_y if i == 0])
while cnt:
    r2 = np.random.randint(len(test_data_y))
    if test_data_y[r2] != 0:
        continue
    r1 = np.random.randint(len(test_data_y))
    test_data_x.insert(r1, test_data_x[r2])
    test_data_y.insert(r1, test_data_y[r2])
    cnt -= 1

with open('/usr/local/oybb/yzx_w2vec/30500000')as f:
    embeddings = json.loads(f.read())


def next():
    r = np.random.randint(0, len(train_data_y), batch_size)
    xx, yy = [], []
    for i in r:
        xx.append(train_data_x[i])
        yy.append(train_data_y[i])
    return xx, yy


x_test, y_test = test_data_x[:batch_size], test_data_y[:batch_size]

x = tf.placeholder(shape=[batch_size, msg_count, msg_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

embeddings = tf.constant(embeddings)
embed = tf.nn.embedding_lookup(embeddings, x)

c1 = ml.conv2d(embed, conv_filter=[1, 4, embedding_size, 1], padding='VALID', ksize=[1, 100, 10, 1],
               pool_stride=[1, 100, 10, 1],
               pool_padding='VALID')
# c2 = ml.conv2d(c1, conv_filter=[4, 4, 1, 1], padding='VALID', ksize=[1, 20, 5, 1],
#                pool_stride=[1, 10, 2, 1],
#                pool_padding='VALID')
c3 = ml.conv2d(c1, conv_filter=[int(c1.shape[1]), int(c1.shape[2]), 1, 1], padding='VALID', ksize=[1, 1, 1, 1],
               pool_stride=[1, 1, 1, 1],
               pool_padding='VALID')

y = tf.nn.sigmoid(ml.layer_basic(c3[:,0,0]) )[:,0]
#gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
# loss=tf.reduce_mean((y-y_)**2)
#l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.2, scope=None), weights_list=gv)
#all_loss = loss + l2_loss
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

import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json

word_size = 1000
embedding_size = 60
batch_size =512
col = pymongo.MongoClient().xingqiao.msg_cut
data = list(col.find({'createTime':{'$lt':'2017/11/5'}}, {'_id': 0, 'status': 1, 'msgCut': 1}))
# data = list(col.find( {},{'_id': 0, 'status': 1, 'msgCut': 1}))

data = [i['msgCut'][:word_size] + [i['status']] for i in data]

train_data = data[:-1 * batch_size]
train_data_neg = [i for i in train_data if i[-1] == 0]
cnt = sum([i[-1] for i in train_data]) - len(train_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(train_data))
    r2 = np.random.randint(len(train_data_neg))
    train_data.insert(r1, train_data_neg[r2])
train_data = pd.DataFrame(train_data)

test_data = data[-1 * batch_size:]
test_data_neg = [i for i in test_data if i[-1] == 0]
cnt = sum([i[-1] for i in test_data]) - len(test_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(test_data))
    r2 = np.random.randint(len(test_data_neg))
    test_data.insert(r1, test_data_neg[r2])
test_data = pd.DataFrame(test_data).sample(batch_size)

with open('/usr/local/oybb/yzx_w2vec/30500000')as f:
    embeddings = json.loads(f.read())


def next():
    a = train_data.sample(batch_size)
    return np.array(a.drop(word_size, axis=1)), np.array(a[word_size])


x_test, y_test = np.array(test_data.drop(word_size, axis=1)), np.array(test_data[word_size])

x = tf.placeholder(shape=[batch_size, word_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

embeddings = tf.constant(embeddings)
embed = tf.nn.embedding_lookup(embeddings, x)
X = tf.reshape(embed, [batch_size, word_size, embedding_size, 1])

c1 = ml.conv2d(X, conv_filter=[10, embedding_size, 1, 2], padding='SAME', ksize=[1, 20, 1, 1], pool_stride=[1, 10, 1, 1],
               pool_padding='SAME')
c2 = ml.conv2d(c1, conv_filter=[4, 1, 2, 4], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 10, 1, 1],
               pool_padding='SAME')
c3 = ml.conv2d(c2, conv_filter=[2, 1, 4, 8], padding='SAME', ksize=[1, 10, 1, 1], pool_stride=[1, 10, 1, 1],
               pool_padding='VALID')
# lay1 = tf.reshape(c2, [batch_size, -1])
# lay2 = ml.layer_basic(ml.bn(lay1), 1)
out=tf.reshape(c3,shape=[batch_size,8])
y = tf.nn.sigmoid(ml.layer_basic(out,1) )[:,0]
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
gv= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#loss=tf.reduce_mean((y-y_)**2)
l2_loss=tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.05, scope=None), weights_list=gv)
all_loss=loss+l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
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

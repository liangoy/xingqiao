import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json
from tensorflow.contrib.rnn import GRUCell

word_size = 100
embedding_size = 8
batch_size =1024
#长度大于10,正反各占一半,!里面有将近一般的短信重复,200以上
with open('/usr/local/oybb/project/xingqiao_data/oneMsg100Gt5000') as f:
    data=json.loads(f.read())

train_data=data[:-8744-1024]
test_data=data[-8744-1024:-8744]

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

embeddings = tf.Variable(
    tf.random_uniform([1000, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x)

gru = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state = gru.zero_state(batch_size, dtype=tf.float32)
lis = []
with tf.variable_scope('RNN'):
    for timestep in range(word_size):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = gru(ml.bn_with_wb(embed[:, timestep]), state)
    out_put = state

lay1 = tf.nn.elu(ml.layer_basic(out_put, 4))
lay2 = ml.layer_basic(out_put, 1)
y = tf.nn.sigmoid(lay2[:, 0])
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
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
              sum(test_y * y_test) / sum (y_test))

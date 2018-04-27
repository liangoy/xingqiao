import tensorflow as tf
import jieba
import pymongo
import pandas as pd
import numpy as np
from tensorflow.contrib.rnn import GRUCell
from util import ml

vocabulary_size = 9027 + 1
word_size = 200
embedding_size = 64
batch_size = 512
col = pymongo.MongoClient().xingqiao.dataWithMsg1000
data = list(col.find({}, {'_id': 0, 'status': 1, 'msg1000': 1}))
data = [i['msg1000'][:word_size] + [i['status']] for i in data]

train_data = data[:-1 * batch_size]
train_data_neg = [i for i in train_data if i[-1] == 0]
cnt = sum([i[-1] for i in train_data]) - len(train_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(train_data))
    r2 = np.random.randint(len(train_data_neg))
    train_data.insert(r1, train_data_neg[r2 - 1])
train_data = pd.DataFrame(train_data)

test_data = data[-1 * batch_size:]
test_data_neg = [i for i in test_data if i[-1] == 0]
cnt = sum([i[-1] for i in test_data]) - len(test_data_neg)
for i in range(cnt):
    r1 = np.random.randint(len(test_data))
    r2 = np.random.randint(len(test_data_neg))
    test_data.insert(r1, test_data_neg[r2 - 1])
test_data = pd.DataFrame(test_data).sample(batch_size)


def next():
    a = train_data.sample(batch_size)
    return np.array(a.drop(word_size, axis=1)), np.array(a[word_size])


x_test, y_test = np.array(test_data.drop(word_size, axis=1)), np.array(test_data[word_size])

x = tf.placeholder(shape=[batch_size, word_size], dtype=tf.int32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

X = x + 1

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, X)

gru = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state = gru.zero_state(batch_size, dtype=tf.float32)
lis = []
with tf.variable_scope('RNN'):
    for timestep in range(word_size):
        if timestep == 1:
            tf.Variable_scope().reuse_variables()
        (cell_output, state) = gru(ml.bn_with_wb(embed[:, timestep]), state)
    out_put = state

lay1 = ml.layer_basic(out_put, 4)
lay2 = ml.layer_basic(out_put, 1)
y = tf.nn.sigmoid(lay2[:, 0])
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................',sum(y_test)/len(y_test))

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

import tensorflow as tf
import pymongo
import pandas as pd
import numpy as np
from util import ml
import json

word_size = 1000
embedding_size = 60
batch_size = 256
col = pymongo.MongoClient().xingqiao.msg_cut
data = list(col.find({'createTime':{'$lt':'2017/11/5'}}, {'_id': 0, 'status': 1, 'msgCut': 1}))

data = [i['msgCut'][:word_size] + [i['status']] for i in data]

train_data = data[:-1 * batch_size]
train_data_neg = [i for i in train_data if i[-1] == 0] * 3

for i in train_data_neg:
    r = np.random.randint(len(train_data))
    train_data.insert(r, i)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(data[-1 * batch_size:])

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

gru = tf.nn.rnn_cell.GRUCell(num_units=16, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state = gru.zero_state(batch_size, dtype=tf.float32)
lis = []
with tf.variable_scope('RNN'):
    for timestep in range(word_size):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = gru(embed[:, timestep], state)
    out_put = state

lay1 = tf.nn.elu(ml.layer_basic(out_put, 4))
lay2 = ml.layer_basic(ml.bn_with_wb(lay1), 1)
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

    if i % 2 == 0:
        train_loss = sess.run(loss, feed_dict={x: x_train, y_: y_train})
        test_loss = sess.run(loss, feed_dict={x: x_test, y_: y_test})
        test_y = sess.run(y, feed_dict={x: x_test, y_: y_test})
        qtest = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size

        train_y = sess.run(y, feed_dict={x: x_train, y_: y_train})
        qtrain = 1 - len([i for i in train_y + y_train if 0.5 <= i <= 1.5]) / batch_size

        print(train_loss, test_loss, qtrain, qtest)
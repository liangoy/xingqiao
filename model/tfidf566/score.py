import tensorflow as tf
import numpy as np
import json
from util import ml
import pymongo

score_size = 1000
batch_size = 3000

db = pymongo.MongoClient().xingqiao
data = db.dataWithMsg.find({}, {'_id': 0, 'msgScore': 1, 'status': 1})
data = [[i['status']] + [np.mean(eval(i['msgScore']))] for i in data if i['msgScore']!='[]']

# =================================================================
test_data_f = np.array([i for i in data[-1 * batch_size:] if i[0] == 0])
test_data_t = np.array([i for i in data[-1 * batch_size:] if i[0] == 1])
r_t = np.random.randint(0, len(test_data_t), int(batch_size / 2))
r_f = np.random.randint(0, len(test_data_f), int(batch_size / 2))
test_data = np.concatenate([test_data_f[r_f], test_data_t[r_t]])
x_test = test_data[:, 1]
y_test = test_data[:, 0]

train_data_f = np.array([i for i in data[:-1 * batch_size] if i[0] == 0])
train_data_t = np.array([i for i in data[:-1 * batch_size] if i[0] == 1])


def next(batch_size=batch_size):
    r_t = np.random.randint(0, len(train_data_t), int(batch_size / 2))
    r_f = np.random.randint(0, len(train_data_f), int(batch_size / 2))
    data = np.concatenate([train_data_t[r_t], train_data_f[r_f]])
    return data[:, 1], data[:, 0]


x = tf.placeholder(shape=[batch_size], dtype=tf.float32)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

X=tf.reshape(x,[batch_size,1])
lay1 = tf.nn.elu(ml.layer_basic(X, 4))
lay2 = ml.layer_basic(lay1, 1)

y = tf.nn.sigmoid(lay2[:, 0])
loss = tf.reduce_sum(-y_ * tf.log(y + 0.000000001) - (1 - y_) * tf.log(1 - y + 0.00000001)) / batch_size / tf.log(2.0)
gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# loss=tf.reduce_mean((y-y_)**2)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.7, scope=None), weights_list=gv)
all_loss = loss + l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................', sum(y_test) / len(y_test))

for i in range(10 ** 10):
    x_train, y_train = next()
    sess.run(optimizer, feed_dict={x: x_train, y_: y_train})

    if i % 100 == 0:
        train_y, train_loss = sess.run((y, loss), feed_dict={x: x_train, y_: y_train})
        test_y, test_loss = sess.run((y, loss), feed_dict={x: x_test, y_: y_test})
        qtest = 1 - len([i for i in test_y + y_test if 0.5 <= i <= 1.5]) / batch_size
        qtrain = 1 - len([i for i in train_y + y_train if 0.5 <= i <= 1.5]) / batch_size
        print(train_loss, test_loss, qtrain, qtest)
